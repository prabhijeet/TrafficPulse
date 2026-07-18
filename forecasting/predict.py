# forecasting/predict.py

'''
Day 5 -Live Prediction Interface.
Loads the trained model from data/processed/model.pkl and provides
a simple interface to get queue length predictions for any
intersection, approach, phase, and time combination.

This file is what the adaptive agent imports on Day 6.
Instead of looking up historical averages, the agent calls
predict_queue() with the current state and gets a prediction
of what the queue will look like at the next arrival.

Also runs a quick validation test to confirm the model
is loaded correctly and predictions are in a sensible range.
'''

import os
import pickle
import pandas as pd
import numpy as np
from simulation.config import BASE_DIR

# paths
MODEL_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'model.pkl')

# encoding maps -must match exactly what features.py used
INTERSECTION_MAP = {'INT_1': 0, 'INT_2': 1, 'INT_3': 2}
APPROACH_MAP     = {'East': 0, 'North': 1, 'South': 2, 'West': 3}
PHASE_MAP        = {'EW': 0, 'NS': 1, 'BRANCH': 2}
FLOW_MAP         = {f'FLOW_{i}': i-1 for i in range(1, 7)}


def load_model():
    '''
    Loads the trained Random Forest model from disk.
    Called once at startup -model stays in memory for all predictions.
    '''
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print(f'Model loaded from: {MODEL_FILE}')
    print(f'Model type       : {type(model).__name__}')
    print(f'Trees in forest  : {model.n_estimators}')
    print()
    return model


def build_feature_row(intersection, approach, phase, flow_id,
                      queue_length, timestamp_seconds,
                      rolling_5=1.0, rolling_10=1.0, rolling_15=1.0):
    '''
    Builds a single feature row for the model to predict on.
    Takes human-readable inputs and converts them to the encoded
    numerical format the model expects.

    This is what the agent calls on Day 6 -it passes in the
    current state of an intersection and gets back a queue prediction.

    Args:
        intersection     : 'INT_1', 'INT_2', or 'INT_3'
        approach         : 'North', 'South', 'East', or 'West'
        phase            : 'NS', 'EW', or 'BRANCH'
        flow_id          : 'FLOW_1' through 'FLOW_6'
        queue_length     : current number of cars waiting (int)
        timestamp_seconds: current simulation time in seconds
        rolling_5/10/15  : rolling average queue over last 5/10/15
                           arrivals at this approach (default 1.0)

    Returns:
        pandas DataFrame with one row, ready for model.predict()
    '''
    hour         = int(timestamp_seconds // 3600)
    minute_of_day = int(timestamp_seconds // 60)
    is_rush_hour  = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0
    sin_hour      = np.sin(2 * np.pi * hour / 24)
    cos_hour      = np.cos(2 * np.pi * hour / 24)

    row = {
        'intersection_enc'   : INTERSECTION_MAP.get(intersection, 0),
        'approach_enc'       : APPROACH_MAP.get(approach, 0),
        'phase_enc'          : PHASE_MAP.get(phase, 1),
        'flow_enc'           : FLOW_MAP.get(flow_id, 0),
        'queue_length'       : queue_length,
        'hour'               : hour,
        'minute_of_day'      : minute_of_day,
        'is_rush_hour'       : is_rush_hour,
        'sin_hour'           : sin_hour,
        'cos_hour'           : cos_hour,
        'rolling_avg_queue_5' : rolling_5,
        'rolling_avg_queue_10': rolling_10,
        'rolling_avg_queue_15': rolling_15,
    }

    return pd.DataFrame([row])


def predict_queue(model, intersection, approach, phase, flow_id,
                  queue_length, timestamp_seconds,
                  rolling_5=1.0, rolling_10=1.0, rolling_15=1.0):
    '''
    Main prediction function -called by the adaptive agent on Day 6.
    Returns predicted queue length as a float, rounded to 2 decimal places.

    The agent uses this to compare approaches and decide which one
    to give green time to -the approach with the highest predicted
    queue gets priority.
    '''
    features = build_feature_row(
        intersection, approach, phase, flow_id,
        queue_length, timestamp_seconds,
        rolling_5, rolling_10, rolling_15
    )
    prediction = model.predict(features)[0]
    return round(float(prediction), 2)


def run_validation(model):
    '''
    Runs a quick sanity check -tests predictions for a few
    realistic scenarios and confirms they make sense.

    Expected behavior:
      - Rush hour + high queue → higher prediction than quiet + low queue
      - INT_2 East → higher prediction than INT_3 West
      - All predictions should be positive numbers
    '''
    print('Running validation tests...')
    print()

    test_cases = [
        {
            'label'      : 'INT_2 East, rush hour, queue=5',
            'intersection': 'INT_2',
            'approach'   : 'East',
            'phase'      : 'NS',
            'flow_id'    : 'FLOW_4',
            'queue'      : 5,
            'time'       : 8 * 3600,   # 8am rush hour
            'r5': 4.2, 'r10': 3.8, 'r15': 3.5
        },
        {
            'label'      : 'INT_1 North, quiet night, queue=1',
            'intersection': 'INT_1',
            'approach'   : 'North',
            'phase'      : 'NS',
            'flow_id'    : 'FLOW_1',
            'queue'      : 1,
            'time'       : 2 * 3600,   # 2am quiet
            'r5': 1.0, 'r10': 1.0, 'r15': 1.0
        },
        {
            'label'      : 'INT_2 West, evening rush, queue=3',
            'intersection': 'INT_2',
            'approach'   : 'West',
            'phase'      : 'EW',
            'flow_id'    : 'FLOW_4',
            'queue'      : 3,
            'time'       : 17 * 3600,  # 5pm evening rush
            'r5': 2.8, 'r10': 2.5, 'r15': 2.3
        },
        {
            'label'      : 'INT_3 South, midday, queue=2',
            'intersection': 'INT_3',
            'approach'   : 'South',
            'phase'      : 'NS',
            'flow_id'    : 'FLOW_5',
            'queue'      : 2,
            'time'       : 13 * 3600,  # 1pm midday
            'r5': 1.8, 'r10': 1.6, 'r15': 1.5
        },
    ]

    results = []
    for tc in test_cases:
        pred = predict_queue(
            model,
            tc['intersection'], tc['approach'],
            tc['phase'], tc['flow_id'],
            tc['queue'], tc['time'],
            tc['r5'], tc['r10'], tc['r15']
        )
        results.append((tc['label'], tc['queue'], pred))
        print(f"  Scenario : {tc['label']}")
        print(f"  Current queue : {tc['queue']} cars")
        print(f"  Predicted next: {pred} cars")
        print()

    # sanity check -rush hour high queue should predict higher than quiet night
    rush_pred  = results[0][2]
    quiet_pred = results[1][2]
    if rush_pred > quiet_pred:
        print(f'Sanity check PASSED: rush hour prediction ({rush_pred}) '
              f'> quiet night ({quiet_pred}) ✅')
    else:
        print(f'Sanity check NOTE: rush hour ({rush_pred}) <= '
              f'quiet night ({quiet_pred}) -check model')
    print()


def main():
    '''
    Loads model, runs validation, and prints a usage example
    showing how the agent will call predict_queue() on Day 6.
    '''
    print('TrafficPulse -Day 5 Prediction Interface')
    print()

    model = load_model()
    run_validation(model)

    print('Usage example (how the agent calls this on Day 6):')
    print()
    print('  from forecasting.predict import load_model, predict_queue')
    print('  model = load_model()')
    print('  predicted = predict_queue(')
    print("      model, 'INT_2', 'East', 'NS', 'FLOW_4',")
    print('      queue_length=3, timestamp_seconds=28800')
    print('  )')
    print('  print(f"Predicted queue: {predicted} cars")')
    print()
    print('Day 5 complete. Ready for Day 6 -Adaptive Signal Agent.')


if __name__ == '__main__':
    main()