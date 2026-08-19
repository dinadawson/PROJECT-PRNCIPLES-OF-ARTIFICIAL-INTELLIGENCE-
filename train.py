"""
train.py
Handles training the model on the training set.
"""


def train_model(model, X_train, y_train):
    """Train (fit) the given model on the training data."""
    print(f"Training on {X_train.shape[0]} rows...")
    model.fit(X_train, y_train)
    print("Training complete.")
    return model
