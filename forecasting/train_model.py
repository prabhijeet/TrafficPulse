# forecasting/train_model.py

'''
Day 5 -Model Training.
Reads features.csv from data/processed/ and trains a Random Forest
regression model to predict queue length at the next arrival event.

Why Random Forest?
  - Handles mixed feature types well (encoded categoricals + continuous)
  - Doesn't need feature scaling (unlike SVM or neural nets)
  - Gives feature importance scores out of the box
  - Resistant to overfitting due to ensemble averaging
  - Fast to train on our dataset size

Steps:
  1. Load features.csv
  2. Split into train (80%) and test (20%)
  3. Train Random Forest
  4. Evaluate on test set (MAE, RMSE, R²)
  5. Plot feature importance
  6. Save model to data/processed/model.pkl
'''

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from simulation.config import BASE_DIR

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# paths
PROCESSED_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
FEATURES_FILE  = os.path.join(PROCESSED_DIR, 'features.csv')
MODEL_FILE     = os.path.join(PROCESSED_DIR, 'model.pkl')
PLOTS_DIR      = os.path.join(BASE_DIR, 'results', 'plots')

# model settings
RANDOM_STATE   = 42
TEST_SIZE      = 0.2      # 80% train, 20% test
N_ESTIMATORS   = 100      # number of trees in the forest
MAX_DEPTH      = 10       # max depth per tree -prevents overfitting

FEATURE_COLS = [
    'intersection_enc',
    'approach_enc',
    'phase_enc',
    'flow_enc',
    'queue_length',
    'hour',
    'minute_of_day',
    'is_rush_hour',
    'sin_hour',
    'cos_hour',
    'rolling_avg_queue_5',
    'rolling_avg_queue_10',
    'rolling_avg_queue_15',
]

TARGET_COL = 'target_queue'


def load_features():
    '''
    Loads the engineered feature matrix from features.csv.
    Separates into X (features) and y (target).
    '''
    df = pd.read_csv(FEATURES_FILE)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print(f'Features loaded : {X.shape[0]:,} rows, {X.shape[1]} features')
    print(f'Target range    : {y.min():.0f} – {y.max():.0f} cars')
    print(f'Target mean     : {y.mean():.2f} cars')
    print()

    return X, y


def split_data(X, y):
    '''
    Splits data into train and test sets.
    shuffle=True randomizes order before splitting -important because
    our data is time-ordered and we don't want the model to just
    memorize the last 20% of the day.
    random_state=42 keeps the split reproducible.
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        shuffle      = True
    )

    print(f'Training rows : {len(X_train):,}')
    print(f'Test rows     : {len(X_test):,}')
    print()

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    '''
    Trains a Random Forest regressor.

    n_estimators=100 means 100 decision trees are built.
    Each tree is trained on a random subset of rows and features.
    The final prediction is the average of all 100 trees.
    Averaging reduces variance and prevents overfitting.

    max_depth=10 limits how deep each tree can grow.
    Without this, trees can memorize the training data perfectly
    but fail to generalize to new data (overfitting).
    '''
    print('Training Random Forest...')

    model = RandomForestRegressor(
        n_estimators = N_ESTIMATORS,
        max_depth    = MAX_DEPTH,
        random_state = RANDOM_STATE,
        n_jobs       = -1    # use all CPU cores for speed
    )

    model.fit(X_train, y_train)
    print('Training complete.')
    print()

    return model


def evaluate_model(model, X_test, y_test):
    '''
    Evaluates the trained model on the held-out test set.

    MAE  (Mean Absolute Error)    -average error in cars
         e.g. MAE=0.3 means predictions are off by 0.3 cars on average
         Lower is better. 0 = perfect.

    RMSE (Root Mean Squared Error) -penalizes large errors more than MAE
         Useful for detecting occasional very wrong predictions.
         Lower is better.

    R²   (R-squared / coefficient of determination)
         How much of the variance in queue length the model explains.
         1.0 = perfect, 0.0 = no better than predicting the mean,
         negative = worse than predicting the mean.
         Higher is better.
    '''
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print('Model performance on test set:')
    print(f'  MAE  : {mae:.4f}  (avg error in cars)')
    print(f'  RMSE : {rmse:.4f}  (penalizes large errors)')
    print(f'  R²   : {r2:.4f}  (1.0 = perfect, 0.0 = baseline)')
    print()

    return y_pred, mae, rmse, r2


def plot_feature_importance(model):
    '''
    Plots which features the Random Forest found most useful.
    Feature importance = how much each feature reduced prediction error
    across all trees, averaged. Higher = more useful.
    This tells us which signals matter most for predicting congestion.
    '''
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(importance.index, importance.values,
                   color='steelblue', alpha=0.8)

    for bar, val in zip(bars, importance.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_title('Random Forest -Feature Importance',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '9_feature_importance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Feature importance plot saved: {path}')


def plot_predictions_vs_actual(y_test, y_pred):
    '''
    Scatter plot of predicted vs actual queue lengths on the test set.
    Perfect predictions would all fall on the diagonal line.
    Points scattered around the line show prediction error.
    '''
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_test, y_pred, alpha=0.3, color='steelblue',
               s=10, label='Predictions')

    # perfect prediction line
    max_val = max(y_test.max(), max(y_pred))
    ax.plot([0, max_val], [0, max_val], 'r--',
            linewidth=2, label='Perfect prediction')

    ax.set_title('Predicted vs Actual Queue Length (Test Set)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Actual Queue Length (cars)')
    ax.set_ylabel('Predicted Queue Length (cars)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '10_predicted_vs_actual.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Predictions vs actual plot saved: {path}')


def save_model(model):
    '''
    Saves the trained model to disk as a .pkl file using pickle.
    pickle serializes the Python object to bytes so it can be
    loaded back later without retraining.
    The adaptive agent on Day 6 loads this file to get predictions.
    '''
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to: {MODEL_FILE}')


def main():
    '''
    Runs the full training pipeline:
      1. Load features
      2. Split train/test
      3. Train Random Forest
      4. Evaluate on test set
      5. Plot feature importance
      6. Plot predictions vs actual
      7. Save model
    '''
    print('TrafficPulse -Day 5 Model Training')
    print()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    X, y                             = load_features()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model                            = train_model(X_train, y_train)
    y_pred, mae, rmse, r2            = evaluate_model(model, X_test, y_test)

    plot_feature_importance(model)
    plot_predictions_vs_actual(y_test, y_pred)
    save_model(model)

    print()
    print(f'Day 5 complete.')
    print(f'Model: {N_ESTIMATORS} trees, max_depth={MAX_DEPTH}')
    print(f'MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}')
    print('Run forecasting/predict.py next to test live predictions.')


if __name__ == '__main__':
    main()