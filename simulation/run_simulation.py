# simulation/run_simulation.py

'''
Entry point for the TrafficPulse simulation.
This is the file you actually run - it calls run_simulation() from
traffic_sim.py, takes the event log it returns, converts it to a
pandas dataframe, and saves it as a CSV to data/logs/traffic_log.csv.

Run this file whenever you want to regenerate the traffic data.
Everything downstream - process mining, ML, the agent, the LLM -
reads from the CSV this file produces.
'''

import os
import pandas as pd
from simulation.traffic_sim import run_simulation
from simulation.config import SIM_DURATION, LOG_DIR, LOG_FILE, LOG_COLUMNS


def save_log(event_log):
    '''
    Takes the raw event log (list of dicts) from the simulation,
    converts it to a pandas dataframe, and saves it as a CSV.
    Creates the data/logs/ folder if it doesn't exist yet.
    '''
    # create the output folder if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # convert list of dicts to a dataframe
    # each dict in event_log becomes one row
    df = pd.DataFrame(event_log, columns=LOG_COLUMNS)

    # save to CSV - index=False means don't write row numbers
    df.to_csv(LOG_FILE, index=False)

    print(f'Log saved to: {LOG_FILE}')
    print(f'Total rows  : {len(df):,}')
    print(f'Columns     : {list(df.columns)}')
    print()
    print('Quick stats:')
    print(df['event_type'].value_counts())
    print()
    print('Events per intersection:')
    print(df['intersection'].value_counts())


def main():
    '''
    Runs the full simulation and saves the output.
    This is what gets called when you run this file directly.
    '''
    print('TrafficPulse - Traffic Digital Twin')
    print(f'Simulating {SIM_DURATION // 3600} hours of traffic...')
    print()

    # run the simulation - returns a list of event dicts
    event_log = run_simulation(SIM_DURATION)

    # save everything to CSV
    save_log(event_log)

    print()
    print('Day 2 complete. Run visualization/plot_queues.py next.')


if __name__ == '__main__':
    main()