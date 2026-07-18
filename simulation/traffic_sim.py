# simulation/traffic_sim.py

'''
The heart of TrafficPulse.
Runs the actual traffic simulation -signals cycle, cars travel through
the network intersection by intersection, queues build and clear, and
every event gets logged to a list that becomes the CSV.

How it works:
  - One signal_controller process per intersection (3 total)
  - One arrival_process per approach per intersection (12 total)
  - Each car is ONE process that travels through multiple intersections
    sequentially -arrive, wait for green, depart, drive to next, repeat
  - Travel time between intersections is randomized (see TRAVEL_TIMES in config)

This version fixes two problems from earlier attempts:
  v1 -polling loop (yield env.timeout(1)) was too slow -millions of events
  v2 -spawning new car processes at each intersection caused infinite chains
  v3 (this version) -one car = one process, travels freely through network
                       with realistic travel time between intersections
                       no spawning chains, no hop limits, richer event log
'''

import simpy
import random
import numpy as np
from simulation.config import (
    INTERSECTIONS, APPROACHES, NETWORK, SIGNAL_PHASES,
    ARRIVAL_RATES, ROUTING, TRAVEL_TIMES, RANDOM_SEED
)


def get_arrival_rate(intersection, approach, current_time_seconds):
    '''
    Returns cars per hour arriving at this approach right now.
    Looks up the correct time window from ARRIVAL_RATES in config.py.
    '''
    current_hour = current_time_seconds / 3600
    rates = ARRIVAL_RATES[intersection][approach]
    for (start, end, rate) in rates:
        if start <= current_hour < end:
            return rate
    return 2


def get_interarrival_time(intersection, approach, current_time_seconds):
    '''
    Returns seconds to wait before the next car arrives.
    Uses exponential distribution -standard model for random arrivals.
    Higher rate = shorter average gap.
    '''
    rate_per_hour   = get_arrival_rate(intersection, approach, current_time_seconds)
    rate_per_second = rate_per_hour / 3600
    return random.expovariate(rate_per_second)


def get_travel_time(from_intersection, to_intersection):
    '''
    Returns a random travel time in seconds between two connected intersections.
    Looks up the range from TRAVEL_TIMES and picks a random value within it.
    Returns 0 if no travel time defined for this pair (shouldn't happen).
    '''
    key = f'{from_intersection}_to_{to_intersection}'
    if key in TRAVEL_TIMES:
        min_t, max_t = TRAVEL_TIMES[key]
        return random.uniform(min_t, max_t)
    return 0


def get_flow_id(intersection, approach, came_from=None):
    '''
    Maps a car to one of the 6 network flows.
    came_from tracks which intersection the car just traveled from.
    Logged in CSV so process mining can reconstruct all 6 flows on Day 4.
    '''
    if intersection == 'INT_1':
        if came_from is None:
            return 'FLOW_1'   # fresh arrival at INT_1 from outside
        else:
            return 'FLOW_2'   # came from INT_2, heading west

    if intersection == 'INT_2':
        if came_from == 'INT_1':
            return 'FLOW_1'   # traveling east through main corridor
        elif came_from == 'INT_3':
            return 'FLOW_3'   # came up from south branch
        else:
            return 'FLOW_4'   # fresh arrival at INT_2 from outside

    if intersection == 'INT_3':
        if came_from is None:
            return 'FLOW_5'   # fresh arrival at INT_3 from outside
        else:
            return 'FLOW_6'   # came down from INT_2, heading south

    return 'FLOW_UNKNOWN'


class IntersectionState:
    '''
    Live scoreboard for one intersection.
    Tracks phase, queues, cars served, and wait times.
    One instance per intersection, shared across all car processes.

    phase_changed is a SimPy Event that fires every time the signal changes.
    Cars subscribe to it instead of polling every second -core optimization.
    '''

    def __init__(self, intersection_id, env):
        self.intersection_id     = intersection_id
        self.env                 = env
        self.current_phase       = SIGNAL_PHASES[intersection_id][0]['name']
        self.current_phase_index = 0
        self.green_approaches    = SIGNAL_PHASES[intersection_id][0]['green_approaches']
        self.queues              = {approach: 0 for approach in APPROACHES}
        self.total_cars_served   = 0
        self.approach_wait_time  = {approach: 0.0 for approach in APPROACHES}
        self.phase_changed       = env.event()

    def update_phase(self, phase_index):
        '''
        Updates phase and fires phase_changed to wake up all waiting cars.
        Creates a fresh event immediately for the next phase change.
        '''
        self.current_phase_index = phase_index
        phase                    = SIGNAL_PHASES[self.intersection_id][phase_index]
        self.current_phase       = phase['name']
        self.green_approaches    = phase['green_approaches']

        old_event          = self.phase_changed
        self.phase_changed = self.env.event()
        old_event.succeed()

    def is_green(self, approach):
        return approach in self.green_approaches


def signal_controller(env, intersection_id, state, event_log):
    '''
    SimPy process -cycles through signal phases forever.
    One instance per intersection. Logs every phase change.
    '''
    phases      = SIGNAL_PHASES[intersection_id]
    phase_index = 0

    while True:
        phase = phases[phase_index]
        state.update_phase(phase_index)

        event_log.append({
            'car_id'      : 'SIGNAL',
            'timestamp'   : round(env.now, 2),
            'hour'        : int(env.now // 3600),
            'intersection': intersection_id,
            'approach'    : 'ALL',
            'event_type'  : 'signal_change',
            'queue_length': sum(state.queues.values()),
            'wait_time'   : 0,
            'phase'       : phase['name'],
            'flow_id'     : 'N/A',
        })

        yield env.timeout(phase['duration'])
        phase_index = (phase_index + 1) % len(phases)


def car_journey(env, car_id, start_intersection, start_approach, states, event_log):
    '''
    SimPy process -one car's complete journey through the network.
    
    One process handles the full trip:
      arrive → wait for green → depart → travel → arrive at next → repeat
    
    Uses a visited set to prevent loops -a car can visit each intersection
    only once. This eliminates the INT_2 ↔ INT_3 loop without cutting
    valid multi-hop journeys short.
    
    The car exits naturally when it hits a network boundary (None in NETWORK)
    or has already visited the next intersection.
    '''
    current_intersection = start_intersection
    current_approach     = start_approach
    came_from            = None
    visited              = set()   # tracks which intersections this car has been to

    while current_intersection is not None:

        # if we've already been here, exit -avoids loops
        if current_intersection in visited:
            break

        visited.add(current_intersection)
        state        = states[current_intersection]
        flow_id      = get_flow_id(current_intersection, current_approach, came_from)
        arrival_time = env.now

        # arrive -join queue and log
        state.queues[current_approach] += 1
        event_log.append({
            'car_id'      : car_id,
            'timestamp'   : round(arrival_time, 2),
            'hour'        : int(arrival_time // 3600),
            'intersection': current_intersection,
            'approach'    : current_approach,
            'event_type'  : 'car_arrived',
            'queue_length': state.queues[current_approach],
            'wait_time'   : 0,
            'phase'       : state.current_phase,
            'flow_id'     : flow_id,
        })

        # wait for green -event driven, not polling
        while not state.is_green(current_approach):
            yield state.phase_changed

        # green -depart
        wait_time = round(env.now - arrival_time, 2)
        state.queues[current_approach] = max(0, state.queues[current_approach] - 1)
        state.total_cars_served += 1

        event_log.append({
            'car_id'      : car_id,
            'timestamp'   : round(env.now, 2),
            'hour'        : int(env.now // 3600),
            'intersection': current_intersection,
            'approach'    : current_approach,
            'event_type'  : 'car_departed',
            'queue_length': state.queues[current_approach],
            'wait_time'   : wait_time,
            'phase'       : state.current_phase,
            'flow_id'     : flow_id,
        })

        # find next intersection
        next_intersection = NETWORK[current_intersection].get(current_approach.lower())

        if next_intersection is None:
            # network boundary -car exits
            break

        if next_intersection in visited:
            # already been there -exit to prevent loop
            break

        # routing at INT_2 -filter out already visited intersections
        direction_key = f'INT_2_from_{current_approach.lower()}'
        if next_intersection == 'INT_2' and direction_key in ROUTING:
            choices = list(ROUTING[direction_key].keys())
            weights = list(ROUTING[direction_key].values())

            # only keep directions that lead to unvisited intersections
            valid         = []
            valid_weights = []
            for choice, weight in zip(choices, weights):
                leads_to = NETWORK['INT_2'].get(choice)
                if leads_to not in visited and leads_to is not None:
                    valid.append(choice)
                    valid_weights.append(weight)

            if not valid:
                # all routes lead to visited intersections -exit
                break

            # renormalize weights
            total         = sum(valid_weights)
            valid_weights = [w / total for w in valid_weights]
            next_approach = np.random.choice(valid, p=valid_weights).capitalize()

        else:
            opposites     = {'North': 'South', 'South': 'North',
                             'East' : 'West',  'West'  : 'East'}
            next_approach = opposites[current_approach]

        # drive to next intersection
        travel_time = get_travel_time(current_intersection, next_intersection)
        yield env.timeout(travel_time)

        # update for next iteration
        came_from            = current_intersection
        current_intersection = next_intersection
        current_approach     = next_approach

        

def arrival_process(env, intersection_id, approach, state, event_log, car_counter, states):
    '''
    SimPy process -generates car arrivals at one approach forever.
    12 of these run simultaneously across all intersections and approaches.
    Each new car gets its own car_journey process.
    '''
    while True:
        gap = get_interarrival_time(intersection_id, approach, env.now)
        yield env.timeout(gap)

        car_counter[0] += 1
        car_id = f'CAR_{car_counter[0]:05d}'

        # spawn one journey process per car -travels freely through network
        env.process(car_journey(
            env, car_id, intersection_id,
            approach, states, event_log
        ))


def run_simulation(sim_duration):
    '''
    Sets up and runs the full simulation.
    Creates states, starts all processes, runs the clock.
    Returns the complete event log as a list of dicts.
    '''
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env         = simpy.Environment()
    event_log   = []
    car_counter = [0]

    states = {i: IntersectionState(i, env) for i in INTERSECTIONS}

    for intersection_id in INTERSECTIONS:
        state = states[intersection_id]
        env.process(signal_controller(env, intersection_id, state, event_log))
        for approach in APPROACHES:
            env.process(arrival_process(
                env, intersection_id, approach,
                state, event_log, car_counter, states
            ))

    print(f'Running simulation for {sim_duration // 3600} hours...')
    env.run(until=sim_duration)
    print(f'Simulation complete. {len(event_log):,} events logged.')
    print(f'Total cars generated: {car_counter[0]:,}')

    return event_log