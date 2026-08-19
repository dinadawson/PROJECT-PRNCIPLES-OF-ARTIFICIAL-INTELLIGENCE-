"""
model.py
Defines the model architecture used in this project: a Random Forest Classifier.
"""

from sklearn.ensemble import RandomForestClassifier
import config


def build_model():
    """Create a fresh (untrained) Random Forest Classifier using the
    hyperparameters defined in config.py."""
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    return model
