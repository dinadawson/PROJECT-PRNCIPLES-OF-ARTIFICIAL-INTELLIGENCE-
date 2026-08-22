"""
config.py
Stores all hyperparameters, file paths, and settings used across the project.
Keeping these in one place means we only need to change values here,
not hunt through every file when we want to tune something.
"""

# ----- Data settings -----
CSV_PATH = "web_page_phishing_dataset.csv"   # local saved copy of the new dataset
KAGGLE_DATASET = "shashwatwork/web-page-phishing-detection-dataset"
URL_COLUMN = "url"           # raw URL text column name in this dataset
LABEL_COLUMN = "status"      # raw label column name (values: "legitimate" / "phishing")
VALIDATION_SIZE = 0.2                  # 20% validation, 80% training
RANDOM_STATE = 42                      # fixed seed so results are reproducible

# ----- Model hyperparameters (Random Forest) -----
N_ESTIMATORS = 200                     # number of trees in the Random Forest
MAX_DEPTH = None                       # let trees grow fully

# ----- Cross-validation / hyperparameter tuning settings -----
CV_FOLDS = 5                           # number of folds for cross-validation
TUNING_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
}

# ----- Investigation settings (data leakage / accuracy checks) -----
TOP_N_FEATURES_TO_ABLATE = 10          # how many top-correlated features to
                                        # remove during the ablation test

# ----- Output file paths -----
CONFUSION_MATRIX_PATH = "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = "feature_importance.png"
MODEL_COMPARISON_PATH = "model_comparison.png"
VALIDATION_PREDICTIONS_PATH = "prediction_results.xlsx"
TRAINING_PREDICTIONS_PATH = "training_prediction_results.xlsx"
FULL_DATASET_EXPORT_PATH = "full_dataset_235795_rows.xlsx"

# ----- Project metadata (for README / report generation) -----
PROJECT_TITLE = "Phishing Website Detection using Random Forest"
DATASET_NAME = "PhiUSIIL Phishing URL Dataset (UCI Machine Learning Repository, 2024)"
TEAM_MEMBERS = {
    "Lead Data Scientist": "Coordination, config, notebook orchestration",
    "Data Scientist I": "Data pipeline: loading, preprocessing, stratified split",
    "Data Scientist II": "Model architecture, training, hyperparameter tuning",
    "Data Scientist III": "Evaluation, visualization, ablation testing",
}


def print_config_summary():
    """Print a readable summary of the current configuration - useful as a
    quick reference when explaining project settings during consultations."""
    print(f"Project: {PROJECT_TITLE}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Validation split: {VALIDATION_SIZE*100:.0f}%")
    print(f"Random Forest: {N_ESTIMATORS} trees, max_depth={MAX_DEPTH}")
    print(f"Cross-validation folds: {CV_FOLDS}")
    print("\nTeam responsibilities:")
    for member, task in TEAM_MEMBERS.items():
        print(f"  - {member}: {task}")
