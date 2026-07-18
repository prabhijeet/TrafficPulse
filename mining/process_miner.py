# mining/process_miner.py

'''
Day 4 -Process Mining.
Reads traffic_log.csv and applies process mining using pm4py.

Process mining reconstructs how the system actually behaves by analyzing
the sequence of events per car journey. We never told pm4py about our
6 flows, signal phases, or network topology -it discovers all of that
automatically from the raw event log.

What this file does:
  1. Loads traffic_log.csv and filters to car events only
  2. Converts it to pm4py's event log format
  3. Discovers the directly-follows graph (process map)
  4. Saves the process map as an image
  5. Prints key statistics about what was discovered

The pm4py format needs 3 things per event:
  case:concept:name  → who is moving (car_id)
  concept:name       → what happened (event_type)
  time:timestamp     → when it happened (timestamp)
'''

import os
import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from simulation.config import BASE_DIR

# paths
LOG_FILE      = os.path.join(BASE_DIR, 'data', 'logs', 'traffic_log.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PLOTS_DIR     = os.path.join(BASE_DIR, 'results', 'plots')


def load_and_filter(filepath):
    '''
    Loads the CSV and filters out signal_change rows.
    Process mining works on car journeys -signals are infrastructure,
    not cases we want to trace. We also drop rows with FLOW_UNKNOWN
    since those can't be reliably assigned to a flow.
    '''
    df = pd.read_csv(filepath)

    # keep only car events
    car_df = df[df['event_type'] != 'signal_change'].copy()

    # drop unknown flows
    car_df = car_df[car_df['flow_id'] != 'FLOW_UNKNOWN'].copy()

    # drop the SIGNAL rows that slipped through (car_id = SIGNAL)
    car_df = car_df[car_df['car_id'] != 'SIGNAL'].copy()

    print(f'Total rows in CSV       : {len(df):,}')
    print(f'Car event rows          : {len(car_df):,}')
    print(f'Unique cars (cases)     : {car_df["car_id"].nunique():,}')
    print(f'Unique flows discovered : {car_df["flow_id"].nunique():,}')
    print()

    return car_df


def convert_to_pm4py_log(car_df):
    '''
    Converts our pandas dataframe to pm4py's internal EventLog format.

    pm4py needs a specific column naming convention:
      case:concept:name  → the ID of the journey (car_id)
      concept:name       → the activity that happened (event_type)
      time:timestamp     → when it happened (as a datetime object)

    We rename our columns to match, convert timestamps from seconds
    to proper datetime objects, then let pm4py's converter do the rest.
    '''
    # rename columns to pm4py convention
    pm_df = car_df.rename(columns={
        'car_id'    : 'case:concept:name',
        'event_type': 'concept:name',
        'timestamp' : 'time:timestamp',
    }).copy()

    # pm4py needs real datetime objects, not raw seconds
    # we treat timestamp=0 as midnight on a fixed reference date
    # and add each car's seconds offset to get a proper datetime
    pm_df['time:timestamp'] = pd.to_datetime(
        pm_df['time:timestamp'], unit='s', origin='2024-01-01'
    )

    # sort by case then timestamp -pm4py requires chronological order
    pm_df = pm_df.sort_values(['case:concept:name', 'time:timestamp'])

    # convert to pm4py EventLog object
    event_log = log_converter.apply(
        pm_df,
        variant=log_converter.Variants.TO_EVENT_LOG
    )

    print(f'pm4py EventLog created  : {len(event_log)} traces')
    print()

    return event_log


def discover_process_map(event_log):
    '''
    Runs the Directly-Follows Graph (DFG) discovery algorithm.
    A DFG shows which activities follow which, and how often.

    In our case the activities are car_arrived and car_departed,
    so the DFG shows the full arrival → departure flow per car
    across all intersections -reconstructed purely from the log.
    '''
    # discover the DFG -returns (graph, start_activities, end_activities)
    dfg, start_activities, end_activities = pm4py.discover_dfg(event_log)

    print('Directly-Follows Graph discovered:')
    print(f'  Edges in graph : {len(dfg)}')
    print(f'  Start activities: {dict(start_activities)}')
    print(f'  End activities  : {dict(end_activities)}')
    print()

    return dfg, start_activities, end_activities


def save_process_map(dfg, start_activities, end_activities):
    '''
    Renders the DFG as an image and saves it to results/plots/.
    Uses pm4py's built-in visualizer -no matplotlib needed here.
    '''
    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, '6_process_map.png')

    # generate the visualization
    gviz = dfg_visualizer.apply(
        dfg,
        parameters={
            dfg_visualizer.Variants.FREQUENCY.value.Parameters.START_ACTIVITIES: start_activities,
            dfg_visualizer.Variants.FREQUENCY.value.Parameters.END_ACTIVITIES  : end_activities,
        }
    )

    # save to file
    dfg_visualizer.save(gviz, output_path)
    print(f'Process map saved to: {output_path}')


def print_flow_summary(car_df):
    '''
    Prints a summary of events per flow so we can verify
    process mining has access to all 6 flows in the log.
    '''
    print('Events per flow:')
    flow_counts = car_df['flow_id'].value_counts().sort_index()
    for flow, count in flow_counts.items():
        print(f'  {flow}: {count:,} events')
    print()

    print('Events per intersection:')
    int_counts = car_df['intersection'].value_counts().sort_index()
    for intersection, count in int_counts.items():
        print(f'  {intersection}: {count:,} events')
    print()

    print('Event types:')
    type_counts = car_df['event_type'].value_counts()
    for etype, count in type_counts.items():
        print(f'  {etype}: {count:,}')
    print()


def save_event_log_csv(car_df):
    '''
    Saves the cleaned, filtered car event log to data/processed/.
    This is the version used by the ML model on Day 5 -signal events
    removed, unknown flows removed, ready for feature engineering.
    '''
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DIR, 'car_events.csv')
    car_df.to_csv(output_path, index=False)
    print(f'Cleaned event log saved to: {output_path}')
    print(f'Rows saved: {len(car_df):,}')
    print()


def main():
    '''
    Runs the full process mining pipeline:
      1. Load and filter the event log
      2. Print flow and intersection summary
      3. Convert to pm4py format
      4. Discover the process map
      5. Save process map image
      6. Save cleaned event log for Day 5
    '''
    print('TrafficPulse -Day 4 Process Mining')
    print('Loading traffic_log.csv...')
    print()

    car_df    = load_and_filter(LOG_FILE)
    print_flow_summary(car_df)
    event_log = convert_to_pm4py_log(car_df)
    dfg, start_activities, end_activities = discover_process_map(event_log)
    save_process_map(dfg, start_activities, end_activities)
    save_event_log_csv(car_df)

    print()
    print('Day 4 complete.')
    print('Check results/plots/6_process_map.png for the process map.')
    print('Run mining/bottleneck.py next for bottleneck analysis.')


if __name__ == '__main__':
    main()