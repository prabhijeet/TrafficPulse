# llm/tools.py

'''
Day 7/8 — LLM Tool Definitions (Complete Set).
All tools the LLM assistant can call to get real data.

Tool-calling prevents hallucination — the LLM calls these Python
functions to get actual data, then uses that data to answer questions.

Tools:
  1. get_intersection_status    — status, queues, wait times per intersection
  2. get_signal_history         — signal phase history
  3. explain_congestion         — what happened at a specific hour
  4. get_agent_comparison       — baseline vs adaptive agent results
  5. get_flow_analysis          — analysis of a specific traffic flow
  6. get_rush_hour_summary      — what happened during rush hours
  7. get_worst_approaches       — top N worst approaches network-wide
  8. get_network_summary        — complete network overview in one call
'''

import os
import json
import pandas as pd
from simulation.config import BASE_DIR

LOG_FILE        = os.path.join(BASE_DIR, 'data', 'logs', 'traffic_log.csv')
PROCESSED_DIR   = os.path.join(BASE_DIR, 'data', 'processed')
BASELINE_FILE   = os.path.join(PROCESSED_DIR, 'baseline_intersection.csv')
ADAPTIVE_FILE   = os.path.join(PROCESSED_DIR, 'adaptive_intersection.csv')
RESULTS_FILE    = os.path.join(BASE_DIR, 'results', 'evaluation_report.txt')

_log_df = None

def _get_log():
    global _log_df
    if _log_df is None:
        _log_df = pd.read_csv(LOG_FILE)
    return _log_df


def get_intersection_status(intersection_id: str) -> str:
    '''
    Returns current status of a specific intersection.
    Includes queue lengths, phase, wait times, total cars served.
    '''
    df    = _get_log()
    valid = ['INT_1', 'INT_2', 'INT_3']
    if intersection_id not in valid:
        return json.dumps({'error': f'Unknown intersection. Valid: {valid}'})

    int_df     = df[df['intersection'] == intersection_id]
    last_hour  = int_df['hour'].max()
    recent     = int_df[int_df['hour'] == last_hour]
    arrivals   = recent[recent['event_type'] == 'car_arrived']
    signals    = recent[recent['event_type'] == 'signal_change']
    departures = int_df[int_df['event_type'] == 'car_departed']

    queue_per_approach = arrivals.groupby('approach')['queue_length'].mean().round(2).to_dict()
    last_phase         = signals['phase'].iloc[-1] if len(signals) > 0 else 'unknown'
    avg_wait           = round(departures['wait_time'].mean(), 2)
    max_wait           = round(departures['wait_time'].max(), 2)
    total_served       = len(departures)
    approach_wait      = departures.groupby('approach')['wait_time'].mean().round(2)
    worst_approach     = approach_wait.idxmax() if len(approach_wait) > 0 else 'N/A'
    worst_wait         = round(approach_wait.max(), 2) if len(approach_wait) > 0 else 0

    return json.dumps({
        'intersection'                   : intersection_id,
        'last_hour'                      : int(last_hour),
        'current_phase'                  : last_phase,
        'queue_per_approach_cars'        : queue_per_approach,
        'avg_wait_time_seconds'          : avg_wait,
        'max_wait_time_seconds'          : max_wait,
        'total_cars_served'              : total_served,
        'worst_approach'                 : worst_approach,
        'worst_approach_avg_wait_seconds': worst_wait,
        'note'                           : 'All wait times in SECONDS'
    }, indent=2)


def get_signal_history(intersection_id: str, last_n_hours: int = 2) -> str:
    '''Returns signal phase history for the last N hours.'''
    df    = _get_log()
    valid = ['INT_1', 'INT_2', 'INT_3']
    if intersection_id not in valid:
        return json.dumps({'error': f'Unknown intersection. Valid: {valid}'})

    int_df     = df[df['intersection'] == intersection_id]
    last_hour  = int(int_df['hour'].max())
    start_hour = max(0, last_hour - last_n_hours + 1)
    signals    = int_df[
        (int_df['event_type'] == 'signal_change') &
        (int_df['hour'] >= start_hour)
    ]

    if len(signals) == 0:
        return json.dumps({'error': 'No signal data found'})

    phase_counts     = signals['phase'].value_counts().to_dict()
    changes_per_hour = {int(k): int(v) for k, v in
                        signals.groupby('hour').size().to_dict().items()}

    return json.dumps({
        'intersection'            : intersection_id,
        'hours_covered'           : f'{start_hour}:00 - {last_hour}:00',
        'total_signal_changes'    : len(signals),
        'phase_frequency'         : phase_counts,
        'changes_per_hour'        : changes_per_hour,
        'avg_queue_at_change_cars': round(signals['queue_length'].mean(), 2),
        'note'                    : 'Queue lengths in cars'
    }, indent=2)


def explain_congestion(intersection_id: str, hour: int) -> str:
    '''Explains congestion at a specific intersection and hour.'''
    df    = _get_log()
    valid = ['INT_1', 'INT_2', 'INT_3']
    if intersection_id not in valid:
        return json.dumps({'error': f'Unknown intersection. Valid: {valid}'})
    if not 0 <= hour <= 23:
        return json.dumps({'error': 'Hour must be 0-23'})

    subset     = df[(df['intersection'] == intersection_id) & (df['hour'] == hour)]
    if len(subset) == 0:
        return json.dumps({'error': f'No data for {intersection_id} at hour {hour}'})

    arrivals   = subset[subset['event_type'] == 'car_arrived']
    departures = subset[subset['event_type'] == 'car_departed']
    signals    = subset[subset['event_type'] == 'signal_change']

    avg_wait   = round(departures['wait_time'].mean(), 2) if len(departures) > 0 else 0
    max_wait   = round(departures['wait_time'].max(), 2) if len(departures) > 0 else 0
    avg_queue  = round(arrivals['queue_length'].mean(), 2) if len(arrivals) > 0 else 0
    max_queue  = int(arrivals['queue_length'].max()) if len(arrivals) > 0 else 0

    if len(departures) > 0:
        approach_wait  = departures.groupby('approach')['wait_time'].mean().round(2)
        worst_approach = approach_wait.idxmax()
        worst_wait_val = round(approach_wait.max(), 2)
    else:
        worst_approach = 'N/A'
        worst_wait_val = 0

    flow_dist  = arrivals['flow_id'].value_counts().to_dict() if len(arrivals) > 0 else {}
    phase_dist = signals['phase'].value_counts().to_dict() if len(signals) > 0 else {}
    is_rush    = 7 <= hour <= 9 or 16 <= hour <= 19

    all_dep    = df[(df['intersection'] == intersection_id) &
                    (df['event_type'] == 'car_departed')]
    day_avg    = round(all_dep['wait_time'].mean(), 2)
    vs_day     = round(avg_wait - day_avg, 2)

    return json.dumps({
        'intersection'               : intersection_id,
        'hour'                       : hour,
        'time_label'                 : f'{hour:02d}:00 - {hour+1:02d}:00',
        'is_rush_hour'               : is_rush,
        'cars_arrived'               : len(arrivals),
        'cars_departed'              : len(departures),
        'avg_wait_time_seconds'      : avg_wait,
        'max_wait_time_seconds'      : max_wait,
        'avg_queue_length_cars'      : avg_queue,
        'max_queue_length_cars'      : max_queue,
        'worst_approach'             : worst_approach,
        'worst_approach_wait_seconds': worst_wait_val,
        'flow_distribution'          : flow_dist,
        'signal_phases'              : phase_dist,
        'vs_day_average_seconds'     : vs_day,
        'congestion_level'           : 'HIGH' if avg_wait > day_avg * 1.2
                                       else 'LOW' if avg_wait < day_avg * 0.8
                                       else 'NORMAL',
        'note'                       : 'All wait times in SECONDS'
    }, indent=2)


def get_agent_comparison(intersection_id: str = 'ALL') -> str:
    '''
    Returns side-by-side comparison of fixed timer vs adaptive agent.
    Shows exact improvement percentages for each intersection.
    Use this when asked whether the agent helped or met its targets.
    '''
    try:
        baseline = pd.read_csv(BASELINE_FILE, index_col=0)
        adaptive = pd.read_csv(ADAPTIVE_FILE, index_col=0)
    except FileNotFoundError:
        return json.dumps({'error': 'Agent comparison data not found. Run agent/evaluate.py first.'})

    intersections = ['INT_1', 'INT_2', 'INT_3'] if intersection_id == 'ALL' \
                    else [intersection_id]

    results = {}
    for i in intersections:
        if i not in baseline.index:
            continue
        b_avg  = round(baseline.loc[i, 'avg_wait_time'], 3)
        a_avg  = round(adaptive.loc[i, 'avg_wait_time'], 3)
        b_max  = round(baseline.loc[i, 'max_wait_time'], 3)
        a_max  = round(adaptive.loc[i, 'max_wait_time'], 3)
        b_cars = int(baseline.loc[i, 'total_served'])
        a_cars = int(adaptive.loc[i, 'total_served'])

        pct_avg  = round(((a_avg - b_avg) / b_avg) * 100, 1)
        pct_max  = round(((a_max - b_max) / b_max) * 100, 1)
        pct_cars = round(((a_cars - b_cars) / b_cars) * 100, 1)

        results[i] = {
            'fixed_timer_avg_wait_seconds'   : b_avg,
            'adaptive_avg_wait_seconds'      : a_avg,
            'avg_wait_improvement_pct'       : pct_avg,
            'fixed_timer_max_wait_seconds'   : b_max,
            'adaptive_max_wait_seconds'      : a_max,
            'max_wait_improvement_pct'       : pct_max,
            'fixed_timer_cars_served'        : b_cars,
            'adaptive_cars_served'           : a_cars,
            'cars_served_improvement_pct'    : pct_cars,
        }

    # overall network summary
    b_overall  = round(baseline['avg_wait_time'].mean(), 3)
    a_overall  = round(adaptive['avg_wait_time'].mean(), 3)
    pct_overall= round(((a_overall - b_overall) / b_overall) * 100, 1)
    b_max_all  = round(baseline['max_wait_time'].max(), 3)
    a_max_all  = round(adaptive['max_wait_time'].max(), 3)
    pct_max_all= round(((a_max_all - b_max_all) / b_max_all) * 100, 1)
    b_total    = int(baseline['total_served'].sum())
    a_total    = int(adaptive['total_served'].sum())

    return json.dumps({
        'per_intersection'                  : results,
        'network_overall_avg_wait_fixed'    : b_overall,
        'network_overall_avg_wait_adaptive' : a_overall,
        'network_avg_wait_improvement_pct'  : pct_overall,
        'network_max_wait_fixed_seconds'    : b_max_all,
        'network_max_wait_adaptive_seconds' : a_max_all,
        'network_max_wait_improvement_pct'  : pct_max_all,
        'total_cars_fixed'                  : b_total,
        'total_cars_adaptive'               : a_total,
        'additional_cars_served'            : a_total - b_total,
        'note'                              : 'All wait times in SECONDS. Negative pct = improvement.'
    }, indent=2)


def get_flow_analysis(flow_id: str) -> str:
    '''
    Returns detailed analysis of a specific traffic flow.
    Shows wait times, volume, peak hours, and which intersections
    the flow passes through.
    Use when asked about a specific flow or route through the network.
    '''
    df    = _get_log()
    valid = [f'FLOW_{i}' for i in range(1, 7)]
    if flow_id not in valid:
        return json.dumps({'error': f'Unknown flow. Valid: {valid}'})

    flow_df    = df[df['flow_id'] == flow_id]
    departures = flow_df[flow_df['event_type'] == 'car_departed']
    arrivals   = flow_df[flow_df['event_type'] == 'car_arrived']

    if len(departures) == 0:
        return json.dumps({'error': f'No departure data for {flow_id}'})

    avg_wait      = round(departures['wait_time'].mean(), 2)
    max_wait      = round(departures['wait_time'].max(), 2)
    total_cars    = len(arrivals)
    intersections = arrivals['intersection'].value_counts().to_dict()

    # peak hour for this flow
    hourly_wait   = departures.groupby('hour')['wait_time'].mean()
    peak_hour     = int(hourly_wait.idxmax())
    peak_wait     = round(hourly_wait.max(), 2)

    # flow route descriptions
    flow_routes = {
        'FLOW_1': 'West → INT_1 → INT_2 → East (main corridor westbound to eastbound)',
        'FLOW_2': 'East → INT_2 → INT_1 → West (main corridor eastbound to westbound)',
        'FLOW_3': 'South → INT_3 → INT_2 → East (south branch joining, going east)',
        'FLOW_4': 'South → INT_3 → INT_2 → INT_1 → West (south branch joining, going west)',
        'FLOW_5': 'West → INT_1 → INT_2 → INT_3 → South (main corridor turning south)',
        'FLOW_6': 'East → INT_2 → INT_3 → South (east turning south)',
    }

    return json.dumps({
        'flow_id'                 : flow_id,
        'route'                   : flow_routes.get(flow_id, 'Unknown'),
        'total_cars'              : total_cars,
        'avg_wait_time_seconds'   : avg_wait,
        'max_wait_time_seconds'   : max_wait,
        'intersections_visited'   : intersections,
        'peak_hour'               : peak_hour,
        'peak_hour_wait_seconds'  : peak_wait,
        'note'                    : 'All wait times in SECONDS'
    }, indent=2)


def get_rush_hour_summary() -> str:
    '''
    Returns what happened during morning and evening rush hours
    across all intersections.
    Use when asked about the morning/evening commute or rush hour performance.
    '''
    df = _get_log()

    summary = {}
    for period, hours, label in [
        ('morning_rush', [7, 8], '07:00-09:00'),
        ('evening_rush', [16, 17, 18], '16:00-19:00'),
    ]:
        period_df  = df[df['hour'].isin(hours)]
        departures = period_df[period_df['event_type'] == 'car_departed']
        arrivals   = period_df[period_df['event_type'] == 'car_arrived']

        by_intersection = {}
        for i in ['INT_1', 'INT_2', 'INT_3']:
            i_dep = departures[departures['intersection'] == i]
            i_arr = arrivals[arrivals['intersection'] == i]
            if len(i_dep) > 0:
                by_intersection[i] = {
                    'cars_arrived'          : len(i_arr),
                    'avg_wait_time_seconds' : round(i_dep['wait_time'].mean(), 2),
                    'max_wait_time_seconds' : round(i_dep['wait_time'].max(), 2),
                    'avg_queue_cars'        : round(i_arr['queue_length'].mean(), 2),
                }

        summary[period] = {
            'time_label'         : label,
            'total_cars_arrived' : len(arrivals),
            'total_cars_served'  : len(departures),
            'by_intersection'    : by_intersection,
        }

    return json.dumps({
        'rush_hour_summary': summary,
        'note'             : 'All wait times in SECONDS'
    }, indent=2)


def get_worst_approaches(top_n: int = 5) -> str:
    '''
    Returns the top N worst performing approaches across the entire network.
    Use when asked which approach or direction has the most delays,
    or which traffic light is causing the most problems.
    '''
    df         = _get_log()
    departures = df[df['event_type'] == 'car_departed'].copy()

    if len(departures) == 0:
        return json.dumps({'error': 'No departure data found'})

    approach_stats = departures.groupby(
        ['intersection', 'approach']
    )['wait_time'].agg(
        avg_wait_seconds = 'mean',
        max_wait_seconds = 'max',
        total_cars       = 'count'
    ).round(2).sort_values('avg_wait_seconds', ascending=False)

    top = approach_stats.head(top_n).reset_index()
    results = []
    for _, row in top.iterrows():
        results.append({
            'rank'                : len(results) + 1,
            'intersection'        : row['intersection'],
            'approach'            : row['approach'],
            'avg_wait_seconds'    : row['avg_wait_seconds'],
            'max_wait_seconds'    : row['max_wait_seconds'],
            'total_cars'          : int(row['total_cars']),
        })

    return json.dumps({
        'top_worst_approaches': results,
        'note'                : 'Ranked by average wait time in SECONDS'
    }, indent=2)


def get_network_summary() -> str:
    '''
    Returns a complete overview of the entire network in one call.
    Includes all intersection stats, total cars, peak hours, and
    flow distribution. Use when asked for a full summary or overview.
    '''
    df         = _get_log()
    departures = df[df['event_type'] == 'car_departed']
    arrivals   = df[df['event_type'] == 'car_arrived']

    by_intersection = {}
    for i in ['INT_1', 'INT_2', 'INT_3']:
        i_dep = departures[departures['intersection'] == i]
        i_arr = arrivals[arrivals['intersection'] == i]
        if len(i_dep) > 0:
            app_wait      = i_dep.groupby('approach')['wait_time'].mean().round(2)
            worst_app     = app_wait.idxmax()
            hourly        = i_dep.groupby('hour')['wait_time'].mean()
            peak_hour     = int(hourly.idxmax())
            by_intersection[i] = {
                'total_cars_served'      : len(i_dep),
                'avg_wait_time_seconds'  : round(i_dep['wait_time'].mean(), 2),
                'max_wait_time_seconds'  : round(i_dep['wait_time'].max(), 2),
                'avg_queue_cars'         : round(i_arr['queue_length'].mean(), 2),
                'worst_approach'         : worst_app,
                'worst_approach_wait_sec': round(app_wait.max(), 2),
                'peak_congestion_hour'   : peak_hour,
            }

    flow_dist = arrivals['flow_id'].value_counts().to_dict()
    peak_hour_overall = int(
        departures.groupby('hour')['wait_time'].mean().idxmax()
    )

    return json.dumps({
        'network_summary'         : {
            'total_cars_arrived'  : len(arrivals),
            'total_cars_served'   : len(departures),
            'simulation_hours'    : 24,
            'peak_congestion_hour': peak_hour_overall,
        },
        'by_intersection'         : by_intersection,
        'flow_distribution'       : flow_dist,
        'note'                    : 'All wait times in SECONDS'
    }, indent=2)


# tool definitions in Anthropic/OpenAI/Groq compatible format
TOOLS = [
    {
        'name'        : 'get_intersection_status',
        'description' : (
            'Returns the current status of a traffic intersection — queue lengths, '
            'active signal phase, average and maximum wait times, total cars served, '
            'and worst performing approach. Use for questions about status, performance, '
            'or wait times of a specific intersection.'
        ),
        'input_schema': {
            'type'      : 'object',
            'properties': {
                'intersection_id': {
                    'type'       : 'string',
                    'description': 'The intersection ID: INT_1, INT_2, or INT_3.',
                    'enum'       : ['INT_1', 'INT_2', 'INT_3']
                }
            },
            'required': ['intersection_id']
        }
    },
    {
        'name'        : 'get_signal_history',
        'description' : (
            'Returns signal phase history for a specific intersection over the last N hours. '
            'Shows phase frequency and signal change counts. '
            'Use for questions about signal timing or phase decisions.'
        ),
        'input_schema': {
            'type'      : 'object',
            'properties': {
                'intersection_id': {
                    'type': 'string',
                    'enum': ['INT_1', 'INT_2', 'INT_3']
                },
                'last_n_hours': {
                    'type'   : 'integer',
                    'default': 2
                }
            },
            'required': ['intersection_id']
        }
    },
    {
        'name'        : 'explain_congestion',
        'description' : (
            'Explains what happened at a specific intersection during a specific hour. '
            'Returns wait times, queue lengths, flow distribution, and congestion level. '
            'Use for questions about why congestion was high at a specific time, '
            'or what happened during morning/evening rush hour at an intersection.'
        ),
        'input_schema': {
            'type'      : 'object',
            'properties': {
                'intersection_id': {
                    'type': 'string',
                    'enum': ['INT_1', 'INT_2', 'INT_3']
                },
                'hour': {
                    'type'       : 'integer',
                    'description': 'Hour of day (0-23). Morning rush = 7-9, Evening rush = 16-18.'
                }
            },
            'required': ['intersection_id', 'hour']
        }
    },
    {
        'name'        : 'get_agent_comparison',
        'description' : (
            'Returns a side-by-side comparison of fixed timer vs adaptive agent results. '
            'Shows exact improvement percentages for wait times and cars served. '
            'Use when asked whether the adaptive agent helped, met its targets, '
            'or how it compares to the fixed timer. Pass ALL for network-wide comparison.'
        ),
        'input_schema': {
            'type'      : 'object',
            'properties': {
                'intersection_id': {
                    'type'       : 'string',
                    'description': 'Intersection to compare, or ALL for full network.',
                    'enum'       : ['INT_1', 'INT_2', 'INT_3', 'ALL']
                }
            },
            'required': ['intersection_id']
        }
    },
    {
        'name'        : 'get_flow_analysis',
        'description' : (
            'Returns detailed analysis of a specific traffic flow including route, '
            'volume, wait times, and peak hours. '
            'Use when asked about a specific flow (FLOW_1 through FLOW_6), '
            'or why a particular route through the network is slow.'
        ),
        'input_schema': {
            'type'      : 'object',
            'properties': {
                'flow_id': {
                    'type'       : 'string',
                    'description': 'Flow ID: FLOW_1 through FLOW_6.',
                    'enum'       : ['FLOW_1', 'FLOW_2', 'FLOW_3',
                                   'FLOW_4', 'FLOW_5', 'FLOW_6']
                }
            },
            'required': ['flow_id']
        }
    },
    {
        'name'        : 'get_rush_hour_summary',
        'description' : (
            'Returns what happened during morning rush (7-9am) and evening rush (4-7pm) '
            'across all intersections. Shows cars, wait times, and queue lengths per period. '
            'Use when asked about the morning commute, evening commute, rush hour traffic, '
            'or how busy it was during peak hours.'
        ),
        'input_schema': {
            'type'      : 'object',
            'properties': {},
            'required'  : []
        }
    },
    {
        'name'        : 'get_worst_approaches',
        'description' : (
            'Returns the top N worst performing approaches across the entire network, '
            'ranked by average wait time. '
            'Use when asked which traffic light, approach, or direction causes the most delays, '
            'or which is the worst bottleneck in the network.'
        ),
        'input_schema': {
            'type'      : 'object',
            'properties': {
                'top_n': {
                    'type'       : 'integer',
                    'description': 'Number of worst approaches to return. Default 5.',
                    'default'    : 5
                }
            },
            'required': []
        }
    },
    {
        'name'        : 'get_network_summary',
        'description' : (
            'Returns a complete overview of the entire network in one call — '
            'all intersection stats, total cars, peak hours, and flow distribution. '
            'Use when asked for a full summary, complete overview, or total network performance. '
            'This is the most efficient tool for broad questions about the whole network.'
        ),
        'input_schema': {
            'type'      : 'object',
            'properties': {},
            'required'  : []
        }
    },
]


def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    '''Routes LLM tool calls to the correct Python function.'''
    dispatch = {
        'get_intersection_status': lambda: get_intersection_status(
            tool_input.get('intersection_id', 'INT_2')
        ),
        'get_signal_history'     : lambda: get_signal_history(
            tool_input.get('intersection_id', 'INT_2'),
            tool_input.get('last_n_hours', 2)
        ),
        'explain_congestion'     : lambda: explain_congestion(
            tool_input.get('intersection_id', 'INT_2'),
            int(tool_input.get('hour', 8))
        ),
        'get_agent_comparison'   : lambda: get_agent_comparison(
            tool_input.get('intersection_id', 'ALL')
        ),
        'get_flow_analysis'      : lambda: get_flow_analysis(
            tool_input.get('flow_id', 'FLOW_4')
        ),
        'get_rush_hour_summary'  : lambda: get_rush_hour_summary(),
        'get_worst_approaches'   : lambda: get_worst_approaches(
            int(tool_input.get('top_n', 5))
        ),
        'get_network_summary'    : lambda: get_network_summary(),
    }

    if tool_name in dispatch:
        return dispatch[tool_name]()
    return json.dumps({'error': f'Unknown tool: {tool_name}'})


if __name__ == '__main__':
    print('Testing all 8 tools...')
    print()

    print('1. get_intersection_status(INT_2):')
    print(get_intersection_status('INT_2'))
    print()

    print('2. get_signal_history(INT_2, 2):')
    print(get_signal_history('INT_2', 2))
    print()

    print('3. explain_congestion(INT_2, 8):')
    print(explain_congestion('INT_2', 8))
    print()

    print('4. get_agent_comparison(ALL):')
    print(get_agent_comparison('ALL'))
    print()

    print('5. get_flow_analysis(FLOW_4):')
    print(get_flow_analysis('FLOW_4'))
    print()

    print('6. get_rush_hour_summary():')
    print(get_rush_hour_summary())
    print()

    print('7. get_worst_approaches(5):')
    print(get_worst_approaches(5))
    print()

    print('8. get_network_summary():')
    print(get_network_summary())
    print()

    print('All 8 tools tested successfully.')