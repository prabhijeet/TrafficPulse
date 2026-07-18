# agent/evaluate.py

'''
Day 6 — Evaluation and Comparison.
Loads baseline (fixed timer) and adaptive agent metrics
from data/processed/ and produces a clean before/after
comparison — the headline result of the entire project.

This is what you present in an interview or demo:
  "The adaptive agent reduced average wait time by X%
   and maximum wait time by Y%, while serving Z more cars."

Generates:
  - Comparison table printed to terminal
  - Plot 11: side by side wait time comparison
  - Plot 12: queue length comparison
  - Plot 13: hourly throughput comparison
  - results/evaluation_report.txt — full written report
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from simulation.config import BASE_DIR

PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PLOTS_DIR     = os.path.join(BASE_DIR, 'results', 'plots')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results')


def load_metrics():
    '''
    Loads both baseline and adaptive metrics from data/processed/.
    Returns paired dataframes for direct comparison.
    '''
    baseline_int  = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'baseline_intersection.csv'),
        index_col=0
    )
    adaptive_int  = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'adaptive_intersection.csv'),
        index_col=0
    )
    baseline_app  = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'baseline_approach.csv'),
        index_col=[0, 1]
    )
    adaptive_app  = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'adaptive_approach.csv'),
        index_col=[0, 1]
    )
    baseline_q    = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'baseline_queue.csv'),
        index_col=0
    )
    adaptive_q    = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'adaptive_queue.csv'),
        index_col=0
    )
    baseline_h    = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'baseline_hourly.csv')
    )
    adaptive_h    = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'adaptive_hourly.csv')
    )

    print('Metrics loaded successfully.')
    print()
    return (baseline_int, adaptive_int, baseline_app, adaptive_app,
            baseline_q, adaptive_q, baseline_h, adaptive_h)


def print_comparison_table(baseline_int, adaptive_int,
                           baseline_q, adaptive_q):
    '''
    Prints the main before/after comparison table.
    Calculates % improvement for each metric.
    '''
    print('=' * 65)
    print('BEFORE vs AFTER — Fixed Timer vs Adaptive Agent')
    print('=' * 65)
    print()

    # per intersection comparison
    print(f'{"Metric":<35} {"Fixed":>8} {"Adaptive":>10} {"Change":>8}')
    print('-' * 65)

    for intersection in ['INT_1', 'INT_2', 'INT_3']:
        b_avg = baseline_int.loc[intersection, 'avg_wait_time']
        a_avg = adaptive_int.loc[intersection, 'avg_wait_time']
        pct   = ((a_avg - b_avg) / b_avg) * 100
        arrow = '✅' if pct < 0 else '⚠️'
        print(f'{intersection} avg wait time{"":<20} '
              f'{b_avg:>8.3f}s {a_avg:>9.3f}s '
              f'{pct:>+7.1f}% {arrow}')

    print('-' * 65)

    # overall metrics
    b_overall = baseline_int['avg_wait_time'].mean()
    a_overall = adaptive_int['avg_wait_time'].mean()
    pct_avg   = ((a_overall - b_overall) / b_overall) * 100

    b_max     = baseline_int['max_wait_time'].max()
    a_max     = adaptive_int['max_wait_time'].max()
    pct_max   = ((a_max - b_max) / b_max) * 100

    b_served  = baseline_int['total_served'].sum()
    a_served  = adaptive_int['total_served'].sum()
    pct_cars  = ((a_served - b_served) / b_served) * 100

    b_avg_q   = baseline_q['avg_queue'].mean()
    a_avg_q   = adaptive_q['avg_queue'].mean()
    pct_queue = ((a_avg_q - b_avg_q) / b_avg_q) * 100

    print(f'{"Overall avg wait time":<35} '
          f'{b_overall:>8.3f}s {a_overall:>9.3f}s '
          f'{pct_avg:>+7.1f}% {"✅" if pct_avg < 0 else "⚠️"}')
    print(f'{"Overall max wait time":<35} '
          f'{b_max:>8.3f}s {a_max:>9.3f}s '
          f'{pct_max:>+7.1f}% {"✅" if pct_max < 0 else "⚠️"}')
    print(f'{"Total cars served":<35} '
          f'{b_served:>8,} {a_served:>10,} '
          f'{pct_cars:>+7.1f}% {"✅" if pct_cars > 0 else "⚠️"}')
    print(f'{"Avg queue length":<35} '
          f'{b_avg_q:>8.3f}  {a_avg_q:>9.3f}  '
          f'{pct_queue:>+7.1f}% {"✅" if pct_queue < 0 else "⚠️"}')

    print()
    print('HEADLINE RESULT:')
    print(f'  Avg wait time reduced by  : {abs(pct_avg):.1f}%')
    print(f'  Max wait time reduced by  : {abs(pct_max):.1f}%')
    print(f'  Additional cars served    : +{a_served - b_served:,}')
    print(f'  INT_2 improvement         : '
          f'{abs(((adaptive_int.loc["INT_2","avg_wait_time"] - baseline_int.loc["INT_2","avg_wait_time"]) / baseline_int.loc["INT_2","avg_wait_time"]) * 100):.1f}%')
    print()

    return pct_avg, pct_max, pct_cars


def plot_wait_time_comparison(baseline_int, adaptive_int):
    '''
    Plot 11 — Side by side bar chart of avg wait time per intersection.
    Fixed timer (blue) vs Adaptive agent (green).
    '''
    intersections = ['INT_1', 'INT_2', 'INT_3']
    x             = np.arange(len(intersections))
    width         = 0.35

    baseline_vals = [baseline_int.loc[i, 'avg_wait_time']
                     for i in intersections]
    adaptive_vals = [adaptive_int.loc[i, 'avg_wait_time']
                     for i in intersections]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1   = ax.bar(x - width/2, baseline_vals, width,
                     label='Fixed Timer', color='steelblue', alpha=0.8)
    bars2   = ax.bar(x + width/2, adaptive_vals, width,
                     label='Adaptive Agent', color='seagreen', alpha=0.8)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.2f}s', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.2f}s', ha='center', fontsize=9)

    ax.set_title('Average Wait Time: Fixed Timer vs Adaptive Agent',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Intersection')
    ax.set_ylabel('Average Wait Time (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels(intersections)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '11_wait_time_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_queue_comparison(baseline_q, adaptive_q):
    '''
    Plot 12 — Side by side bar chart of avg queue length per intersection.
    '''
    intersections = ['INT_1', 'INT_2', 'INT_3']
    x             = np.arange(len(intersections))
    width         = 0.35

    baseline_vals = [baseline_q.loc[i, 'avg_queue']
                     for i in intersections]
    adaptive_vals = [adaptive_q.loc[i, 'avg_queue']
                     for i in intersections]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, baseline_vals, width,
           label='Fixed Timer', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, adaptive_vals, width,
           label='Adaptive Agent', color='seagreen', alpha=0.8)

    ax.set_title('Average Queue Length: Fixed Timer vs Adaptive Agent',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Intersection')
    ax.set_ylabel('Average Queue Length (cars)')
    ax.set_xticks(x)
    ax.set_xticklabels(intersections)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '12_queue_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_hourly_throughput(baseline_h, adaptive_h):
    '''
    Plot 13 — Hourly throughput comparison across the full day.
    Shows whether the adaptive agent serves more cars per hour.
    '''
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(baseline_h['hour'], baseline_h['cars_served'],
            label='Fixed Timer', color='steelblue',
            linewidth=2, marker='o', markersize=4)
    ax.plot(adaptive_h['hour'], adaptive_h['cars_served'],
            label='Adaptive Agent', color='seagreen',
            linewidth=2, marker='s', markersize=4)

    ax.fill_between(baseline_h['hour'], baseline_h['cars_served'],
                    adaptive_h['cars_served'],
                    where=adaptive_h['cars_served'] >= baseline_h['cars_served'],
                    alpha=0.2, color='seagreen', label='Agent advantage')

    ax.set_title('Hourly Cars Served: Fixed Timer vs Adaptive Agent',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Cars Served')
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '13_hourly_throughput.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def save_evaluation_report(pct_avg, pct_max, pct_cars,
                           baseline_int, adaptive_int):
    '''
    Saves a plain text evaluation report to results/.
    This is what you'd attach to a research paper or portfolio.
    '''
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, 'evaluation_report.txt')

    b_int2 = baseline_int.loc['INT_2', 'avg_wait_time']
    a_int2 = adaptive_int.loc['INT_2', 'avg_wait_time']
    pct_int2 = ((a_int2 - b_int2) / b_int2) * 100

    with open(report_path, 'w') as f:
        f.write('TrafficPulse — Evaluation Report\n')
        f.write('=' * 50 + '\n\n')
        f.write('SUMMARY\n')
        f.write('-' * 50 + '\n')
        f.write(f'Average wait time improvement : {abs(pct_avg):.1f}%\n')
        f.write(f'Maximum wait time improvement : {abs(pct_max):.1f}%\n')
        f.write(f'Additional cars served        : +{pct_cars:.1f}%\n')
        f.write(f'INT_2 bottleneck improvement  : {abs(pct_int2):.1f}%\n\n')
        f.write('METHODOLOGY\n')
        f.write('-' * 50 + '\n')
        f.write('Both agents ran on identical simulation conditions:\n')
        f.write('  Same random seed (42)\n')
        f.write('  Same arrival rates\n')
        f.write('  Same network topology\n')
        f.write('  Same simulation duration (24 hours)\n\n')
        f.write('Fixed Timer: 30s green per phase, fixed cycle\n')
        f.write('Adaptive Agent: ML-informed phase selection\n')
        f.write('  Random Forest (100 trees, max_depth=10)\n')
        f.write('  4-level tie-breaking policy\n')
        f.write('  Green time: base + load bonus (max 10s extra)\n\n')
        f.write('INTERSECTION RESULTS\n')
        f.write('-' * 50 + '\n')
        for i in ['INT_1', 'INT_2', 'INT_3']:
            b = baseline_int.loc[i, 'avg_wait_time']
            a = adaptive_int.loc[i, 'avg_wait_time']
            p = ((a - b) / b) * 100
            f.write(f'{i}: {b:.3f}s → {a:.3f}s ({p:+.1f}%)\n')

    print(f'Evaluation report saved: {report_path}')


def main():
    '''
    Loads metrics, prints comparison table,
    generates 3 plots, saves report.
    '''
    print('TrafficPulse — Day 6 Evaluation')
    print()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    (baseline_int, adaptive_int,
     baseline_app, adaptive_app,
     baseline_q,   adaptive_q,
     baseline_h,   adaptive_h) = load_metrics()

    pct_avg, pct_max, pct_cars = print_comparison_table(
        baseline_int, adaptive_int, baseline_q, adaptive_q
    )

    plot_wait_time_comparison(baseline_int, adaptive_int)
    plot_queue_comparison(baseline_q, adaptive_q)
    plot_hourly_throughput(baseline_h, adaptive_h)

    save_evaluation_report(pct_avg, pct_max, pct_cars,
                           baseline_int, adaptive_int)

    print()
    print('Day 6 evaluation complete.')
    print('Open results/plots/ to see plots 11, 12, 13.')
    print('Open results/evaluation_report.txt for full report.')


if __name__ == '__main__':
    main()