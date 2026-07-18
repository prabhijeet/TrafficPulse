# agent/adaptive_agent.py

'''
Day 6 - Adaptive Signal Agent.
Replaces the fixed timer with an ML-informed decision maker.

Instead of blindly cycling through phases on a fixed schedule,
this agent:
  1. Reads current queue lengths at each approach
  2. Uses the trained Random Forest to predict future queue lengths
  3. Allocates green time proportionally to predicted congestion
  4. Applies the 4-level tie-breaking policy when queues are equal
  5. Respects minimum and maximum green time constraints

The agent re-runs the simulation with its own signal controller
replacing the fixed timer. Everything else stays identical —
same cars, same arrival rates, same random seed - so the
comparison is fair.

4-Level Tie-Breaking Policy:
  Level 1 - INT_2 gets structural priority (highest downstream impact)
  Level 2 - Longest waiting approach gets priority (fairness)
  Level 3 - Rush hour directional bias (domain knowledge)
  Level 4 - Round-robin rotation (absolute last resort)
'''

import os
import random
import numpy as np
import simpy
import pandas as pd
from simulation.config import (
    INTERSECTIONS, APPROACHES, NETWORK, SIGNAL_PHASES,
    ARRIVAL_RATES, ROUTING, TRAVEL_TIMES,
    RANDOM_SEED, BASE_DIR, MIN_GREEN, MAX_GREEN, INT2_BRANCH_MIN
)
from forecasting.predict import load_model, predict_queue

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# how many cars the agent looks back at for rolling averages
ROLLING_WINDOW = 10


class AdaptiveIntersectionState:
    '''
    Extended version of IntersectionState from traffic_sim.py.
    Adds rolling queue history per approach so the agent can
    compute rolling averages for the ML model's features.
    Also tracks time_since_last_served for tie-breaking Level 2.
    '''

    def __init__(self, intersection_id, env):
        self.intersection_id    = intersection_id
        self.env                = env
        self.current_phase      = SIGNAL_PHASES[intersection_id][0]['name']
        self.current_phase_index= 0
        self.green_approaches   = SIGNAL_PHASES[intersection_id][0]['green_approaches']
        self.queues             = {a: 0 for a in APPROACHES}
        self.total_cars_served  = 0
        self.phase_changed      = env.event()

        # rolling queue history per approach (last N arrivals)
        self.queue_history      = {a: [] for a in APPROACHES}

        # tracks how long each approach has been waiting
        # used for tie-breaking Level 2
        self.last_served_time   = {a: 0.0 for a in APPROACHES}

    def update_phase(self, phase_index, phase_name, green_approaches):
        '''Updates phase and fires phase_changed event.'''
        self.current_phase_index = phase_index
        self.current_phase       = phase_name
        self.green_approaches    = green_approaches
        old_event                = self.phase_changed
        self.phase_changed       = self.env.event()
        old_event.succeed()

    def is_green(self, approach):
        return approach in self.green_approaches

    def record_arrival(self, approach, queue_length):
        '''Adds queue_length to rolling history for this approach.'''
        self.queue_history[approach].append(queue_length)
        if len(self.queue_history[approach]) > ROLLING_WINDOW * 2:
            self.queue_history[approach].pop(0)

    def get_rolling_avg(self, approach, window):
        '''Returns rolling average queue over last N arrivals.'''
        history = self.queue_history[approach]
        if not history:
            return 1.0
        recent = history[-window:] if len(history) >= window else history
        return round(np.mean(recent), 3)

    def mark_served(self, approach):
        '''Records when this approach last got a green light.'''
        self.last_served_time[approach] = self.env.now

    def time_waiting(self, approach):
        '''Returns how long since this approach last got green.'''
        return self.env.now - self.last_served_time[approach]


def get_predicted_queues(model, state, env):
    '''
    Gets ML predictions for all approaches at this intersection.
    Returns a dict of {approach: predicted_queue_length}.
    Used by the agent to decide which phase to prioritize.
    '''
    predictions = {}
    for approach in APPROACHES:
        pred = predict_queue(
            model,
            state.intersection_id,
            approach,
            state.current_phase,
            'FLOW_1',   # simplified - agent uses queue prediction not flow
            state.queues[approach],
            env.now,
            rolling_5  = state.get_rolling_avg(approach, 5),
            rolling_10 = state.get_rolling_avg(approach, 10),
            rolling_15 = state.get_rolling_avg(approach, 15),
        )
        predictions[approach] = pred
    return predictions


def choose_next_phase(intersection_id, state, model, env,
                      last_phase_index, phases):
    '''
    Core decision function - picks the next signal phase.
    Uses predicted queue lengths to allocate green time.
    Applies 4-level tie-breaking when predictions are equal.

    Returns: (phase_index, green_duration)
    '''
    predictions = get_predicted_queues(model, state, env)
    current_hour = int(env.now // 3600)

    # map each phase to its total predicted queue load
    phase_scores = {}
    for i, phase in enumerate(phases):
        score = sum(
            predictions.get(a, 1.0)
            for a in phase['green_approaches']
        )
        phase_scores[i] = score

    max_score = max(phase_scores.values())
    tied = [i for i, s in phase_scores.items()
            if abs(s - max_score) < 0.01]

    if len(tied) == 1:
        chosen_index = tied[0]
    else:
        # 4-level tie-breaking
        # Level 1 - INT_2 structural priority
        if intersection_id == 'INT_2':
            branch_idx = next(
                (i for i, p in enumerate(phases) if p['name'] == 'BRANCH'),
                None
            )
            ns_idx = next(
                (i for i, p in enumerate(phases) if p['name'] == 'NS'),
                None
            )
            if branch_idx in tied:
                chosen_index = branch_idx
            elif ns_idx in tied:
                chosen_index = ns_idx
            else:
                chosen_index = tied[0]

        # Level 2 - longest waiting approach
        else:
            best_wait = -1
            chosen_index = tied[0]
            for i in tied:
                phase_wait = max(
                    state.time_waiting(a)
                    for a in phases[i]['green_approaches']
                )
                if phase_wait > best_wait:
                    best_wait    = phase_wait
                    chosen_index = i

        # Level 3 - rush hour directional bias
        if len(tied) > 1:
            if 7 <= current_hour <= 9:
                ew_idx = next(
                    (i for i, p in enumerate(phases)
                     if p['name'] == 'EW' and i in tied), None
                )
                if ew_idx is not None:
                    chosen_index = ew_idx
            elif 16 <= current_hour <= 19:
                ns_idx = next(
                    (i for i, p in enumerate(phases)
                     if p['name'] == 'NS' and i in tied), None
                )
                if ns_idx is not None:
                    chosen_index = ns_idx

        # Level 4 - round robin (last resort)
        if chosen_index == last_phase_index and len(tied) > 1:
            remaining = [i for i in tied if i != last_phase_index]
            if remaining:
                chosen_index = remaining[0]

    # calculate green duration proportional to predicted load
    chosen_phase  = phases[chosen_index]
    phase_score   = phase_scores[chosen_index]
    total_score   = sum(phase_scores.values()) or 1

  # keep duration close to the original fixed phase duration
    # but give a small bonus (up to 10s extra) to heavily loaded phases
    # this avoids over-correcting and starving other approaches
    base_duration = chosen_phase['duration']   # original fixed duration (30s)
    base_min      = INT2_BRANCH_MIN if chosen_phase['name'] == 'BRANCH' else MIN_GREEN

    # bonus proportional to how much busier this phase is vs average
    avg_score = sum(phase_scores.values()) / len(phase_scores)
    if avg_score > 0:
        load_ratio = phase_score / avg_score
        bonus      = min(10, (load_ratio - 1) * 5)   # max 10s bonus
        bonus      = max(0, bonus)                     # no negative bonus
    else:
        bonus = 0

    duration = round(
        min(MAX_GREEN, max(base_min, base_duration + bonus)), 1
    )

    return chosen_index, duration


def adaptive_signal_controller(env, intersection_id, state,
                                model, event_log):
    '''
    SimPy process - adaptive signal controller.
    Replaces the fixed timer signal_controller from traffic_sim.py.
    Runs forever, picking the best phase at each cycle using
    ML predictions and the 4-level tie-breaking policy.
    '''
    phases      = SIGNAL_PHASES[intersection_id]
    phase_index = 0

    while True:
        phase    = phases[phase_index]
        duration = phase['duration']   # will be overridden by agent below

        # agent picks next phase and duration
        phase_index, duration = choose_next_phase(
            intersection_id, state, model, env, phase_index, phases
        )
        phase = phases[phase_index]

        state.update_phase(phase_index, phase['name'],
                           phase['green_approaches'])

        # mark all green approaches as served
        for approach in phase['green_approaches']:
            state.mark_served(approach)

        event_log.append({
            'car_id'      : 'SIGNAL',
            'timestamp'   : round(env.now, 2),
            'hour'        : int(env.now // 3600),
            'intersection': intersection_id,
            'approach'    : 'ALL',
            'event_type'  : 'signal_change',
            'queue_length': sum(state.queues.values()),
            'wait_time'   : duration,
            'phase'       : phase['name'],
            'flow_id'     : 'N/A',
        })

        yield env.timeout(duration)


def get_arrival_rate(intersection, approach, current_time_seconds):
    '''Returns cars per hour for current time window.'''
    current_hour = current_time_seconds / 3600
    rates        = ARRIVAL_RATES[intersection][approach]
    for (start, end, rate) in rates:
        if start <= current_hour < end:
            return rate
    return 2


def get_interarrival_time(intersection, approach, current_time_seconds):
    '''Returns random gap before next car arrival.'''
    rate_per_hour   = get_arrival_rate(intersection, approach,
                                       current_time_seconds)
    rate_per_second = rate_per_hour / 3600
    return random.expovariate(rate_per_second)


def get_travel_time(from_int, to_int):
    '''Returns random travel time between intersections.'''
    key = f'{from_int}_to_{to_int}'
    if key in TRAVEL_TIMES:
        mn, mx = TRAVEL_TIMES[key]
        return random.uniform(mn, mx)
    return 0


def car_journey(env, car_id, start_intersection, start_approach,
                states, event_log):
    '''
    Same car journey logic as traffic_sim.py v3.
    One process per car, travels freely through network,
    visited set prevents loops.
    '''
    current_intersection = start_intersection
    current_approach     = start_approach
    came_from            = None
    visited              = set()

    while current_intersection is not None:
        if current_intersection in visited:
            break
        visited.add(current_intersection)

        state        = states[current_intersection]
        arrival_time = env.now

        state.queues[current_approach] += 1
        state.record_arrival(current_approach,
                             state.queues[current_approach])

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
            'flow_id'     : 'N/A',
        })

        while not state.is_green(current_approach):
            yield state.phase_changed

        wait_time = round(env.now - arrival_time, 2)
        state.queues[current_approach] = max(
            0, state.queues[current_approach] - 1
        )
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
            'flow_id'     : 'N/A',
        })

        next_intersection = NETWORK[current_intersection].get(
            current_approach.lower()
        )
        if next_intersection is None or next_intersection in visited:
            break

        direction_key = f'INT_2_from_{current_approach.lower()}'
        if next_intersection == 'INT_2' and direction_key in ROUTING:
            choices  = list(ROUTING[direction_key].keys())
            weights  = list(ROUTING[direction_key].values())
            valid    = [(c, w) for c, w in zip(choices, weights)
                        if NETWORK['INT_2'].get(c) not in visited]
            if not valid:
                break
            choices, weights = zip(*valid)
            total   = sum(weights)
            weights = [w / total for w in weights]
            next_approach = np.random.choice(choices, p=weights).capitalize()
        else:
            opposites     = {'North': 'South', 'South': 'North',
                             'East' : 'West',  'West'  : 'East'}
            next_approach = opposites[current_approach]

        travel_time = get_travel_time(current_intersection,
                                      next_intersection)
        yield env.timeout(travel_time)

        came_from            = current_intersection
        current_intersection = next_intersection
        current_approach     = next_approach


def arrival_process(env, intersection_id, approach, state,
                    event_log, car_counter, states):
    '''Generates car arrivals at one approach forever.'''
    while True:
        gap = get_interarrival_time(intersection_id, approach, env.now)
        yield env.timeout(gap)
        car_counter[0] += 1
        car_id = f'CAR_{car_counter[0]:05d}'
        env.process(car_journey(
            env, car_id, intersection_id,
            approach, states, event_log
        ))


def run_adaptive_simulation(sim_duration, model):
    '''
    Runs the full simulation with the adaptive agent
    replacing the fixed timer signal controller.
    Everything else is identical to the baseline run.
    '''
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env         = simpy.Environment()
    event_log   = []
    car_counter = [0]

    states = {i: AdaptiveIntersectionState(i, env)
              for i in INTERSECTIONS}

    for intersection_id in INTERSECTIONS:
        state = states[intersection_id]
        # adaptive controller replaces fixed timer here
        env.process(adaptive_signal_controller(
            env, intersection_id, state, model, event_log
        ))
        for approach in APPROACHES:
            env.process(arrival_process(
                env, intersection_id, approach,
                state, event_log, car_counter, states
            ))

    print(f'Running adaptive simulation for '
          f'{sim_duration // 3600} hours...')
    env.run(until=sim_duration)
    print(f'Adaptive simulation complete.')
    print(f'Events logged    : {len(event_log):,}')
    print(f'Cars generated   : {car_counter[0]:,}')
    print()

    return event_log


def compute_adaptive_metrics(event_log):
    '''
    Computes the same metrics as fixed_timer.py
    so evaluate.py can do a direct comparison.
    '''
    df         = pd.DataFrame(event_log)
    cars       = df[df['event_type'] != 'signal_change']
    departures = cars[cars['event_type'] == 'car_departed'].copy()
    arrivals   = cars[cars['event_type'] == 'car_arrived'].copy()

    int_metrics = departures.groupby('intersection').agg(
        avg_wait_time = ('wait_time', 'mean'),
        max_wait_time = ('wait_time', 'max'),
        total_served  = ('wait_time', 'count'),
        std_wait_time = ('wait_time', 'std')
    ).round(3)

    approach_metrics = departures.groupby(
        ['intersection', 'approach']
    ).agg(
        avg_wait_time = ('wait_time', 'mean'),
        max_wait_time = ('wait_time', 'max'),
        total_served  = ('wait_time', 'count')
    ).round(3).sort_values('avg_wait_time', ascending=False)

    queue_metrics = arrivals.groupby('intersection').agg(
        avg_queue = ('queue_length', 'mean'),
        max_queue = ('queue_length', 'max')
    ).round(3)

    departures['hour'] = departures['hour'].astype(int)
    hourly = departures.groupby('hour').size().reset_index(
        name='cars_served'
    )

    return int_metrics, approach_metrics, queue_metrics, hourly


def save_adaptive_metrics(int_metrics, approach_metrics,
                          queue_metrics, hourly):
    '''Saves adaptive agent metrics for comparison in evaluate.py.'''
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    int_metrics.to_csv(
        os.path.join(PROCESSED_DIR, 'adaptive_intersection.csv')
    )
    approach_metrics.to_csv(
        os.path.join(PROCESSED_DIR, 'adaptive_approach.csv')
    )
    queue_metrics.to_csv(
        os.path.join(PROCESSED_DIR, 'adaptive_queue.csv')
    )
    hourly.to_csv(
        os.path.join(PROCESSED_DIR, 'adaptive_hourly.csv'),
        index=False
    )
    print('Adaptive metrics saved to data/processed/')


def main():
    '''
    Loads the ML model, runs the adaptive simulation,
    computes metrics, saves for comparison.
    '''
    from simulation.config import SIM_DURATION

    print('TrafficPulse - Day 6 Adaptive Signal Agent')
    print()

    model = load_model()

    event_log = run_adaptive_simulation(SIM_DURATION, model)

    int_metrics, approach_metrics, queue_metrics, hourly = \
        compute_adaptive_metrics(event_log)

    print('Adaptive Agent Results:')
    print(int_metrics.to_string())
    print()

    overall_avg = int_metrics['avg_wait_time'].mean()
    overall_max = int_metrics['max_wait_time'].max()
    total       = int_metrics['total_served'].sum()

    print('HEADLINE METRICS (Adaptive Agent):')
    print(f'  Overall avg wait time : {overall_avg:.3f} seconds')
    print(f'  Overall max wait time : {overall_max:.3f} seconds')
    print(f'  Total cars served     : {total:,}')
    print()

    save_adaptive_metrics(int_metrics, approach_metrics,
                          queue_metrics, hourly)

    print()
    print('Adaptive agent complete. Run agent/evaluate.py next.')


if __name__ == '__main__':
    main()