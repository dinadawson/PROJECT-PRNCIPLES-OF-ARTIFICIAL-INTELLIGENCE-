"""
data_pipeline.py
Handles everything related to getting the data ready for training:
- downloading/loading the dataset
- cleaning it
- splitting it fairly into train/validation sets
- a sanity check to confirm there's no data leakage
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def download_and_save_dataset():
    """Download the PhiUSIIL dataset from UCI ONCE and save it locally as CSV.
    If the CSV already exists, this does nothing (so we never hit the website
    unnecessarily on future runs)."""
    if os.path.exists(config.CSV_PATH):
        print(f"{config.CSV_PATH} already exists — skipping download.")
        return

    from ucimlrepo import fetch_ucirepo
    print("Downloading dataset from UCI (first time only)...")
    phiusiil = fetch_ucirepo(id=config.DATASET_ID)
    X = phiusiil.data.features.copy()
    y = phiusiil.data.targets.copy()

    full_df = X.copy()
    full_df["label"] = y.values.ravel()
    full_df.to_csv(config.CSV_PATH, index=False)
    print(f"Saved dataset to {config.CSV_PATH} ({full_df.shape[0]} rows)")


def load_dataset():
    """Load the dataset from the local CSV file (not from the live website)."""
    df = pd.read_csv(config.CSV_PATH)
    return df


def preprocess(df):
    """Clean the raw dataframe:
    - keep the URL text aside for display purposes only
    - drop non-numeric columns (the model can't use raw text)
    - separate features (X) from label (y)

    Returns: df_numeric, url_text, X, y
    """
    url_text = df["URL"].copy() if "URL" in df.columns else None

    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    print(f"Dropping non-numeric columns: {non_numeric_cols}")

    df_numeric = df.drop(columns=non_numeric_cols)

    missing_count = df_numeric.isnull().sum().sum()
    print(f"Total missing values: {missing_count}")

    X = df_numeric.drop(columns=["label"])
    y = df_numeric["label"]

    return df_numeric, url_text, X, y


def stratified_split(df_numeric, X, y):
    """Split the data fairly per class (phishing / legitimate) then merge.
    This guarantees both classes are proportionally represented in both
    the training set and the validation set.

    Returns: X_train, X_val, y_train, y_val, train_idx, val_idx
    """
    legit_idx = df_numeric[df_numeric["label"] == 1].index
    legit_train_idx, legit_val_idx = train_test_split(
        legit_idx, test_size=config.VALIDATION_SIZE, random_state=config.RANDOM_STATE
    )

    phish_idx = df_numeric[df_numeric["label"] == 0].index
    phish_train_idx, phish_val_idx = train_test_split(
        phish_idx, test_size=config.VALIDATION_SIZE, random_state=config.RANDOM_STATE
    )

    train_idx = legit_train_idx.union(phish_train_idx)
    val_idx = legit_val_idx.union(phish_val_idx)

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]

    print(f"Train set: {len(train_idx)} rows, Validation set: {len(val_idx)} rows")

    return X_train, X_val, y_train, y_val, train_idx, val_idx


def check_no_leakage(train_idx, val_idx):
    """Sanity check: confirm no row appears in both train and validation sets."""
    overlap = train_idx.intersection(val_idx)
    print(f"Overlapping rows between train and validation: {len(overlap)}")
    if len(overlap) == 0:
        print("No data leakage detected.")
    else:
        print("WARNING: Data leakage detected!")
    return len(overlap) == 0
