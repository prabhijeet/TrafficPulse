# visualization/plot_queues.py

'''
Day 3 -Visualization and Validation.
Reads traffic_log.csv and generates 5 plots that verify the simulation
produced realistic traffic patterns. These plots are our sanity check
before we move to process mining and ML.

Plots generated:
  1. Queue length over 24 hours per intersection
  2. Average wait time by hour
  3. Cars arriving per hour (rush hour peaks)
  4. Event distribution across intersections
  5. Flow distribution across all 6 flows

All plots saved to results/plots/
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from simulation.config import BASE_DIR

# paths
LOG_FILE  = os.path.join(BASE_DIR, 'data', 'logs', 'traffic_log.csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')


def load_data():
    '''
    Reads the CSV and splits it into two dataframes —
    one for car events only, one for signal events only.
    Filters out signal rows for most plots since we care about car behavior.
    '''
    df         = pd.read_csv(LOG_FILE)
    car_df     = df[df['event_type'] != 'signal_change'].copy()
    signal_df  = df[df['event_type'] == 'signal_change'].copy()
    print(f'Total rows loaded : {len(df):,}')
    print(f'Car event rows    : {len(car_df):,}')
    print(f'Signal event rows : {len(signal_df):,}')
    print()
    return df, car_df, signal_df


def plot_arrivals_per_hour(car_df):
    '''
    Plot 1 -Cars arriving per hour across the full day.
    This is the most important validation plot -we should clearly
    see morning rush (hours 7-8) and evening rush (hours 16-18).
    '''
    arrivals = car_df[car_df['event_type'] == 'car_arrived']
    hourly   = arrivals.groupby('hour').size()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(hourly.index, hourly.values, color='steelblue', alpha=0.8)

    # highlight rush hours in a different color
    rush_hours = [7, 8, 16, 17, 18]
    for bar, hour in zip(bars, hourly.index):
        if hour in rush_hours:
            bar.set_color('tomato')

    ax.set_title('Cars Arriving Per Hour -Full Day', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Car Arrivals')
    ax.set_xticks(range(24))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x):02d}:00'
    ))
    ax.tick_params(axis='x', rotation=45)

    # add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2,
                str(int(height)), ha='center', va='bottom', fontsize=8)

    ax.legend(handles=[
        plt.Rectangle((0,0),1,1, color='tomato',    label='Rush hour'),
        plt.Rectangle((0,0),1,1, color='steelblue', label='Normal hour'),
    ])

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '1_arrivals_per_hour.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_queue_length_per_intersection(car_df):
    '''
    Plot 2 -Average queue length per hour per intersection.
    INT_2 should show the highest queues since all flows pass through it.
    Rush hour peaks should be clearly visible.
    '''
    arrivals = car_df[car_df['event_type'] == 'car_arrived']

    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {'INT_1': 'steelblue', 'INT_2': 'tomato', 'INT_3': 'seagreen'}

    for intersection in ['INT_1', 'INT_2', 'INT_3']:
        subset  = arrivals[arrivals['intersection'] == intersection]
        hourly  = subset.groupby('hour')['queue_length'].mean()
        ax.plot(hourly.index, hourly.values,
                label=intersection,
                color=colors[intersection],
                linewidth=2, marker='o', markersize=4)

    ax.set_title('Average Queue Length Per Hour by Intersection',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Queue Length (cars)')
    ax.set_xticks(range(24))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x):02d}:00'
    ))
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '2_queue_length_per_intersection.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_average_wait_time(car_df):
    '''
    Plot 3 -Average wait time per hour across all intersections.
    Wait time = seconds a car spent waiting at a red light.
    Higher during rush hours when queues are longer.
    '''
    departures = car_df[car_df['event_type'] == 'car_departed']
    hourly     = departures.groupby('hour')['wait_time'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(hourly.index, hourly.values, alpha=0.4, color='purple')
    ax.plot(hourly.index, hourly.values, color='purple', linewidth=2)

    ax.set_title('Average Wait Time Per Hour (All Intersections)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Wait Time (seconds)')
    ax.set_xticks(range(24))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x):02d}:00'
    ))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '3_average_wait_time.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_events_per_intersection(df):
    '''
    Plot 4 -Total events per intersection as a pie chart.
    INT_2 should dominate since all 6 flows pass through it.
    Validates our network design is working as intended.
    '''
    counts = df.groupby('intersection').size()
    # exclude signal-only label
    counts = counts[counts.index.isin(['INT_1', 'INT_2', 'INT_3'])]

    fig, ax = plt.subplots(figsize=(7, 7))
    colors  = ['steelblue', 'tomato', 'seagreen']
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    for text in autotexts:
        text.set_fontsize(12)

    ax.set_title('Total Events Per Intersection',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '4_events_per_intersection.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_flow_distribution(car_df):
    '''
    Plot 5 -Event count per flow ID.
    All 6 flows should be present. FLOW_1 and FLOW_4 dominate
    because they're the main corridor entries.
    This validates that the routing logic is producing all 6 flows.
    '''
    flow_counts = car_df['flow_id'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(flow_counts.index, flow_counts.values,
                  color='steelblue', alpha=0.8)

    ax.set_title('Event Distribution Across 6 Traffic Flows',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Flow ID')
    ax.set_ylabel('Number of Events')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 5,
                str(int(height)), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '5_flow_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def main():
    '''
    Runs all 5 plots in sequence.
    Creates the plots directory if it doesn't exist.
    '''
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print('TrafficPulse -Day 3 Visualization')
    print('Loading traffic_log.csv...')
    print()

    df, car_df, signal_df = load_data()

    print('Generating plots...')
    print()
    plot_arrivals_per_hour(car_df)
    plot_queue_length_per_intersection(car_df)
    plot_average_wait_time(car_df)
    plot_events_per_intersection(df)
    plot_flow_distribution(car_df)

    print()
    print('All 5 plots saved to results/plots/')
    print('Open that folder in Finder to view them.')


if __name__ == '__main__':
    main()