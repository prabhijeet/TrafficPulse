# simulation/config.py

'''
This is the central config file for TrafficPulse.
Every setting lives here - network layout, signal timing, arrival rates,
routing probabilities. All other files just import from this one.
If something needs changing, this is the only place you touch.
'''

import os

# Simulation runs for one full day. 24 * 3600 = 86400 seconds.
# RANDOM_SEED keeps results reproducible across runs - useful when
# comparing the fixed timer vs the adaptive agent later on Day 6.
SIM_DURATION = 24 * 3600
RANDOM_SEED  = 42

'''
Network layout - T-shaped, 3 intersections, all roads bidirectional:

  West → [INT_1] ←──────────→ [INT_2] → East
                                  ↕
                               [INT_3]
                                  ↕
                                South

None means that direction hits the network boundary - cars exit there.
'''

INTERSECTIONS = ['INT_1', 'INT_2', 'INT_3']
APPROACHES    = ['North', 'South', 'East', 'West']

NETWORK = {
    'INT_1': {
        'east':  'INT_2',   # road to INT_2
        'west':  None,      # network edge, cars exit here
        'north': None,      # local approach only
        'south': None       # local approach only
    },
    'INT_2': {
        'west':  'INT_1',   # road back to INT_1
        'east':  None,      # network edge, cars exit here
        'south': 'INT_3',   # road down to INT_3
        'north': None       # network edge, cars exit here
    },
    'INT_3': {
        'north': 'INT_2',   # road up to INT_2
        'south': None,      # network edge, cars exit here
        'east':  None,      # local approach only
        'west':  None       # local approach only
    }
}

'''
Signal phases per intersection.
INT_1 and INT_3 run a simple 2-phase cycle - NS green, then EW green.
INT_2 needs 3 phases because traffic hits it from 3 directions at once.
duration is in seconds. green_approaches tells the sim which lanes get green.
'''

SIGNAL_PHASES = {
    'INT_1': [
        {'name': 'NS', 'duration': 30, 'green_approaches': ['North', 'South']},
        {'name': 'EW', 'duration': 30, 'green_approaches': ['East',  'West' ]},
    ],
    'INT_2': [
        {'name': 'NS',     'duration': 30, 'green_approaches': ['North', 'South']},
        {'name': 'EW',     'duration': 30, 'green_approaches': ['East',  'West' ]},
        {'name': 'BRANCH', 'duration': 20, 'green_approaches': ['South', 'North']},
    ],
    'INT_3': [
        {'name': 'NS', 'duration': 30, 'green_approaches': ['North', 'South']},
        {'name': 'EW', 'duration': 30, 'green_approaches': ['East',  'West' ]},
    ]
}



# These limits are used by the adaptive agent on Day 6.
# MIN_GREEN stops pedestrians getting caught mid-crossing.
# MAX_GREEN stops one approach hogging the signal the whole time.
MIN_GREEN       = 10
MAX_GREEN       = 90
INT2_BRANCH_MIN = 15   # branch phase at INT_2 needs a slightly higher minimum

'''
Routing probabilities at INT_2.
INT_2 is the only place where cars have to pick a direction.
Which way they go depends on where they came from.
Each dict must sum to 1.0.
'''

ROUTING = {
    'INT_2_from_west':  {'east': 0.6, 'south': 0.4},   # came via INT_1
    'INT_2_from_south': {'east': 0.5, 'west':  0.5},   # came via INT_3
    'INT_2_from_east':  {'west': 0.6, 'south': 0.4},   # entered from east boundary

}

# Travel time between connected intersections in seconds.
# Represents the time a car spends driving between two intersections
# after departing one and before arriving at the next.
# Values are (min_seconds, max_seconds) - actual time is randomized
# within this range using uniform distribution.
TRAVEL_TIMES = {
    'INT_1_to_INT_2': (45, 90),   # ~1 min drive between INT_1 and INT_2
    'INT_2_to_INT_1': (45, 90),   # same road, opposite direction
    'INT_2_to_INT_3': (30, 60),   # slightly shorter south branch road
    'INT_3_to_INT_2': (30, 60),   # same road, opposite direction
}

'''
Arrival rates - how many cars per hour show up at each approach.
Format: (start_hour, end_hour, cars_per_hour)
Morning rush is 7-9am, evening rush is 4-7pm (16-19).
INT_2 gets the highest numbers since all 6 flows pass through it.
'''

# Arrival rates increased 3-5x from previous version to generate
# richer queue distributions for ML training.
# Target: queues of 3-8 cars during rush hour instead of always 1.
ARRIVAL_RATES = {
    'INT_1': {
        'North': [(0,7,30),  (7,9,180), (9,16,75),  (16,19,150), (19,24,30)],
        'South': [(0,7,30),  (7,9,150), (9,16,75),  (16,19,120), (19,24,30)],
        'East':  [(0,7,25),  (7,9,120), (9,16,60),  (16,19,105), (19,24,25)],
        'West':  [(0,7,30),  (7,9,210), (9,16,90),  (16,19,180), (19,24,30)],
    },
    'INT_2': {
        'North': [(0,7,30),  (7,9,150), (9,16,75),  (16,19,135), (19,24,30)],
        'South': [(0,7,30),  (7,9,120), (9,16,60),  (16,19,105), (19,24,30)],
        'East':  [(0,7,36),  (7,9,270), (9,16,120), (16,19,240), (19,24,36)],
        'West':  [(0,7,36),  (7,9,240), (9,16,105), (16,19,210), (19,24,36)],
    },
    'INT_3': {
        'North': [(0,7,25),  (7,9,120), (9,16,60),  (16,19,105), (19,24,25)],
        'South': [(0,7,25),  (7,9,150), (9,16,75),  (16,19,120), (19,24,25)],
        'East':  [(0,7,15),  (7,9,75),  (9,16,45),  (16,19,60),  (19,24,15)],
        'West':  [(0,7,15),  (7,9,75),  (9,16,45),  (16,19,60),  (19,24,15)],
    }
}

# BASE_DIR walks two levels up from this file to get to the project root.
# This way paths work no matter where you run the script from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR  = os.path.join(BASE_DIR, 'data', 'logs')
LOG_FILE = os.path.join(LOG_DIR,  'traffic_log.csv')

# Column order for the CSV. Don't rename these without updating traffic_sim.py.
LOG_COLUMNS = [
    'car_id',        # unique ID per car, e.g. CAR_0001
    'timestamp',     # seconds since simulation start
    'hour',          # 0-23, derived from timestamp // 3600
    'intersection',  # INT_1, INT_2, or INT_3
    'approach',      # North / South / East / West
    'event_type',    # car_arrived, signal_change, or car_departed
    'queue_length',  # number of cars waiting at this approach when event fired
    'wait_time',     # 0 for arrivals, actual seconds waited for departures
    'phase',         # active signal phase at the time: NS, EW, or BRANCH
    'flow_id',       # which of the 6 network flows this car belongs to
]