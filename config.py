"""
config.py
Stores all hyperparameters, file paths, and settings used across the project.
Keeping these in one place means we only need to change values here,
not hunt through every file when we want to tune something.
"""

# ----- Data settings -----
DATASET_ID = 967                       # UCI repo ID for PhiUSIIL Phishing URL Dataset
CSV_PATH = "phiusiil_dataset.csv"      # local saved copy of the dataset
VALIDATION_SIZE = 0.2                  # 20% validation, 80% training
RANDOM_STATE = 42                      # fixed seed so results are reproducible

# ----- Model hyperparameters -----
N_ESTIMATORS = 200                     # number of trees in the Random Forest
MAX_DEPTH = None                       # let trees grow fully

# ----- Output file paths -----
CONFUSION_MATRIX_PATH = "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = "feature_importance.png"
VALIDATION_PREDICTIONS_PATH = "prediction_results.xlsx"
TRAINING_PREDICTIONS_PATH = "training_prediction_results.xlsx"
FULL_DATASET_EXPORT_PATH = "full_dataset_235795_rows.xlsx"
