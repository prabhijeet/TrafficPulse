# mining/bottleneck.py

'''
Day 4 -Bottleneck Analysis.
Reads the cleaned car_events.csv from data/processed/ and identifies
where the worst congestion is happening in the network.

Process mining told us the overall flow pattern. This file goes deeper —
it answers specific operational questions:
  - Which intersection has the longest average wait times?
  - Which approach direction is the worst bottleneck?
  - Which hour of day has peak congestion per intersection?
  - How do the 6 flows compare in terms of wait time?

All findings saved to data/processed/bottleneck_report.csv
and printed as a clean summary report.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from simulation.config import BASE_DIR

# paths
PROCESSED_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
CAR_EVENTS     = os.path.join(PROCESSED_DIR, 'car_events.csv')
REPORT_FILE    = os.path.join(PROCESSED_DIR, 'bottleneck_report.csv')
PLOTS_DIR      = os.path.join(BASE_DIR, 'results', 'plots')


def load_data():
    '''
    Loads car_events.csv -the cleaned event log from process_miner.py.
    Splits into arrivals and departures since wait_time only exists
    on departure rows (it's 0 on arrival rows by design).
    '''
    df         = pd.read_csv(CAR_EVENTS)
    arrivals   = df[df['event_type'] == 'car_arrived'].copy()
    departures = df[df['event_type'] == 'car_departed'].copy()

    print(f'Car events loaded  : {len(df):,}')
    print(f'Arrival rows       : {len(arrivals):,}')
    print(f'Departure rows     : {len(departures):,}')
    print()

    return df, arrivals, departures


def analyse_by_intersection(departures):
    '''
    Calculates average and maximum wait time per intersection.
    The intersection with the highest average wait time is the bottleneck.
    Returns a dataframe sorted worst to best.
    '''
    stats = departures.groupby('intersection')['wait_time'].agg(
        avg_wait  = 'mean',
        max_wait  = 'max',
        total_cars= 'count',
        std_wait  = 'std'
    ).round(2).sort_values('avg_wait', ascending=False)

    print('Wait time by intersection (worst first):')
    print(stats.to_string())
    print()
    print(f'>>> BOTTLENECK: {stats.index[0]} '
          f'(avg wait: {stats["avg_wait"].iloc[0]:.2f}s)')
    print()

    return stats


def analyse_by_approach(departures):
    '''
    Breaks down wait time by intersection + approach direction.
    Tells us not just which intersection is worst but which specific
    direction cars are waiting longest at.
    '''
    stats = departures.groupby(['intersection', 'approach'])['wait_time'].agg(
        avg_wait  = 'mean',
        max_wait  = 'max',
        total_cars= 'count'
    ).round(2).sort_values('avg_wait', ascending=False)

    print('Wait time by intersection + approach (top 10 worst):')
    print(stats.head(10).to_string())
    print()

    return stats


def analyse_by_hour(departures):
    '''
    Shows how congestion changes throughout the day per intersection.
    Peak hours should match our rush hour design (7-9am, 4-7pm).
    '''
    stats = departures.groupby(['intersection', 'hour'])['wait_time'].agg(
        avg_wait  = 'mean',
        total_cars= 'count'
    ).round(2)

    print('Peak congestion hours per intersection:')
    for intersection in ['INT_1', 'INT_2', 'INT_3']:
        subset    = stats.loc[intersection] if intersection in stats.index else None
        if subset is not None:
            peak_hour = subset['avg_wait'].idxmax()
            peak_wait = subset['avg_wait'].max()
            print(f'  {intersection}: peak at hour {peak_hour:02d}:00 '
                  f'(avg wait: {peak_wait:.2f}s)')
    print()

    return stats


def analyse_by_flow(departures):
    '''
    Compares wait times across all 6 traffic flows.
    Flows that pass through INT_2 should have higher wait times
    since INT_2 has a longer signal cycle (3 phases vs 2).
    '''
    stats = departures.groupby('flow_id')['wait_time'].agg(
        avg_wait  = 'mean',
        max_wait  = 'max',
        total_cars= 'count'
    ).round(2).sort_values('avg_wait', ascending=False)

    print('Wait time by flow (worst first):')
    print(stats.to_string())
    print()

    return stats


def plot_bottleneck_heatmap(departures):
    '''
    Plots a heatmap of average wait time per intersection per hour.
    Rows = intersections, Columns = hours.
    Dark red = high wait time = congestion hotspot.
    This is the most visually useful bottleneck plot for presentations.
    '''
    pivot = departures.pivot_table(
        values  = 'wait_time',
        index   = 'intersection',
        columns = 'hour',
        aggfunc = 'mean'
    ).round(2)

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{h:02d}:00' for h in pivot.columns],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title('Average Wait Time Heatmap -Intersection × Hour of Day',
                 fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Avg Wait Time (seconds)')
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, '7_bottleneck_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_wait_time_by_flow(departures):
    '''
    Bar chart comparing average wait time across all 6 flows.
    Shows which routes through the network are slowest.
    '''
    flow_stats = departures.groupby('flow_id')['wait_time'].mean().round(2)
    flow_stats = flow_stats.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(flow_stats.index, flow_stats.values,
                  color='tomato', alpha=0.8)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

    ax.set_title('Average Wait Time by Traffic Flow',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Flow ID')
    ax.set_ylabel('Average Wait Time (seconds)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '8_wait_time_by_flow.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def save_bottleneck_report(int_stats, approach_stats, hour_stats, flow_stats):
    '''
    Saves all 4 analysis results as separate sheets in one CSV report.
    Uses a simple separator row between sections so it's readable in Excel.
    This is what you'd present in a research paper or interview demo.
    '''
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with open(REPORT_FILE, 'w') as f:
        f.write('TRAFFICPULSE BOTTLENECK REPORT\n\n')

        f.write('=== WAIT TIME BY INTERSECTION ===\n')
        int_stats.to_csv(f)
        f.write('\n')

        f.write('=== WAIT TIME BY APPROACH ===\n')
        approach_stats.to_csv(f)
        f.write('\n')

        f.write('=== WAIT TIME BY FLOW ===\n')
        flow_stats.to_csv(f)
        f.write('\n')

    print(f'Bottleneck report saved to: {REPORT_FILE}')


def main():
    '''
    Runs the full bottleneck analysis pipeline.
    Loads cleaned data, runs 4 analyses, generates 2 plots,
    saves the report.
    '''
    print('TrafficPulse -Day 4 Bottleneck Analysis')
    print()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    df, arrivals, departures = load_data()

    int_stats      = analyse_by_intersection(departures)
    approach_stats = analyse_by_approach(departures)
    hour_stats     = analyse_by_hour(departures)
    flow_stats     = analyse_by_flow(departures)

    plot_bottleneck_heatmap(departures)
    plot_wait_time_by_flow(departures)

    save_bottleneck_report(int_stats, approach_stats, hour_stats, flow_stats)

    print()
    print('Day 4 bottleneck analysis complete.')
    print('Open results/plots/ to see plots 7 and 8.')


if __name__ == '__main__':
    main()