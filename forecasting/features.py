# forecasting/features.py

'''
Day 5 -Feature Engineering.
Reads car_events.csv from data/processed/ and builds the feature
matrix the ML model will train on.

Feature engineering is the process of turning raw event log data
into meaningful numerical inputs the model can learn from.
Raw data alone (timestamps, event types) isn't enough -we need
to extract patterns like "how busy was this intersection 10 minutes
ago?" or "is it currently rush hour?".

Features built here:
  - Time-based features (hour, minute, is_rush_hour)
  - Queue-based features (current queue, rolling averages)
  - Intersection and approach encoded as numbers
  - Signal phase encoded as a number
  - Flow ID encoded as a number

Target variable:
  queue_length at the NEXT arrival event for the same
  intersection+approach combination -i.e. how busy will
  this approach be 1 event from now.

Output saved to data/processed/features.csv
'''

import os
import pandas as pd
import numpy as np
from simulation.config import BASE_DIR

# paths
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
CAR_EVENTS    = os.path.join(PROCESSED_DIR, 'car_events.csv')
FEATURES_FILE = os.path.join(PROCESSED_DIR, 'features.csv')


def load_arrivals():
    '''
    Loads car_events.csv and keeps only car_arrived rows.
    We build features on arrivals because that's when queue_length
    is recorded -how many cars were already waiting when this car
    showed up. Departure rows have queue_length after the car left,
    which is less useful for predicting future congestion.
    '''
    df       = pd.read_csv(CAR_EVENTS)
    arrivals = df[df['event_type'] == 'car_arrived'].copy()
    arrivals = arrivals.sort_values('timestamp').reset_index(drop=True)

    print(f'Total car events loaded : {len(df):,}')
    print(f'Arrival rows kept       : {len(arrivals):,}')
    print()

    return arrivals


def encode_categoricals(df):
    '''
    Converts categorical columns to numbers so the ML model can use them.
    ML models work with numbers, not strings. We use simple integer encoding
    since our categories are small and well-defined.

    intersection: INT_1=0, INT_2=1, INT_3=2
    approach:     East=0, North=1, South=2, West=3
    phase:        EW=0, NS=1, BRANCH=2
    flow_id:      FLOW_1=0 ... FLOW_6=5
    '''
    df = df.copy()

    intersection_map = {'INT_1': 0, 'INT_2': 1, 'INT_3': 2}
    approach_map     = {'East': 0, 'North': 1, 'South': 2, 'West': 3}
    phase_map        = {'EW': 0, 'NS': 1, 'BRANCH': 2}
    flow_map         = {f'FLOW_{i}': i-1 for i in range(1, 7)}

    df['intersection_enc'] = df['intersection'].map(intersection_map)
    df['approach_enc']     = df['approach'].map(approach_map)
    df['phase_enc']        = df['phase'].map(phase_map)
    df['flow_enc']         = df['flow_id'].map(flow_map)

    return df


def add_time_features(df):
    '''
    Extracts time-based features from the timestamp column.
    timestamp is in seconds since simulation start (midnight = 0).

    hour         : 0-23, captures time of day pattern
    minute_of_day: 0-1439, finer granularity than hour alone
    is_rush_hour : 1 if morning (7-9) or evening (16-19) rush, else 0
    sin_hour     : sine encoding of hour -captures cyclical nature
    cos_hour     : cosine encoding of hour -pairs with sin_hour
                   (sin/cos encoding means hour 23 and hour 0 are
                   recognized as close together, not far apart)
    '''
    df = df.copy()

    df['hour']          = (df['timestamp'] // 3600).astype(int)
    df['minute_of_day'] = (df['timestamp'] // 60).astype(int)
    df['is_rush_hour']  = df['hour'].apply(
        lambda h: 1 if (7 <= h <= 9 or 16 <= h <= 19) else 0
    )

    # cyclical encoding -treats hour as a circle not a line
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df


def add_rolling_features(df):
    '''
    Adds rolling average queue lengths per intersection+approach.
    Rolling averages capture the recent trend -"has this approach
    been getting busier over the last few arrivals?"

    We calculate rolling mean over the last 5, 10, and 15 arrivals
    for each intersection+approach combination separately.
    min_periods=1 means we still get a value even at the start
    when there aren't enough rows yet.
    '''
    df = df.copy()

    for window in [5, 10, 15]:
        col_name = f'rolling_avg_queue_{window}'
        df[col_name] = (
            df.groupby(['intersection', 'approach'])['queue_length']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
            .round(3)
        )

    return df


def add_target(df):
    '''
    Creates the target variable -what we want the model to predict.

    Target: queue_length at the NEXT arrival for the same
    intersection+approach. This is a short-horizon forecast —
    "given everything happening right now, how long will the
    queue be when the next car arrives here?"

    We use shift(-1) within each intersection+approach group
    to look one event ahead. The last row per group gets NaN
    (no next event) and gets dropped.
    '''
    df = df.copy()

    df['target_queue'] = (
        df.groupby(['intersection', 'approach'])['queue_length']
        .shift(-1)
    )

    # drop rows where target is NaN (last event per group)
    before = len(df)
    df     = df.dropna(subset=['target_queue']).reset_index(drop=True)
    after  = len(df)

    print(f'Rows before dropping NaN targets : {before:,}')
    print(f'Rows after dropping NaN targets  : {after:,}')
    print(f'Rows dropped                     : {before - after:,}')
    print()

    return df


def select_features(df):
    '''
    Selects the final feature columns and target column.
    Drops raw string columns and any columns not needed for training.
    Returns X (feature matrix) and y (target vector) separately.
    '''
    feature_cols = [
        'intersection_enc',      # which intersection (0,1,2)
        'approach_enc',          # which direction (0,1,2,3)
        'phase_enc',             # current signal phase (0,1,2)
        'flow_enc',              # which flow (0-5)
        'queue_length',          # current queue length
        'hour',                  # hour of day (0-23)
        'minute_of_day',         # minute of day (0-1439)
        'is_rush_hour',          # rush hour flag (0 or 1)
        'sin_hour',              # cyclical hour encoding
        'cos_hour',              # cyclical hour encoding
        'rolling_avg_queue_5',   # rolling avg last 5 arrivals
        'rolling_avg_queue_10',  # rolling avg last 10 arrivals
        'rolling_avg_queue_15',  # rolling avg last 15 arrivals
    ]

    X = df[feature_cols]
    y = df['target_queue']

    print(f'Feature matrix shape : {X.shape}')
    print(f'Target vector shape  : {y.shape}')
    print(f'Features used        : {len(feature_cols)}')
    print()
    print('Feature columns:')
    for col in feature_cols:
        print(f'  {col}')
    print()

    return X, y, feature_cols


def save_features(df):
    '''
    Saves the fully engineered feature dataframe to data/processed/.
    train_model.py reads this file directly -it doesn't need to
    redo all the feature engineering steps.
    '''
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(FEATURES_FILE, index=False)
    print(f'Features saved to: {FEATURES_FILE}')
    print(f'Total rows       : {len(df):,}')
    print(f'Total columns    : {len(df.columns):,}')


def main():
    '''
    Runs the full feature engineering pipeline:
      1. Load arrival events
      2. Encode categorical columns
      3. Add time features
      4. Add rolling queue features
      5. Add target variable
      6. Select and verify final features
      7. Save to CSV
    '''
    print('TrafficPulse -Day 5 Feature Engineering')
    print()

    df = load_arrivals()
    df = encode_categoricals(df)
    df = add_time_features(df)
    df = add_rolling_features(df)
    df = add_target(df)

    X, y, feature_cols = select_features(df)
    save_features(df)

    print()
    print('Feature engineering complete.')
    print('Run forecasting/train_model.py next.')


if __name__ == '__main__':
    main()