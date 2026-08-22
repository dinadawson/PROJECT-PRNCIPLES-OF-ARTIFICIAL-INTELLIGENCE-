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
    """Download the Web Page Phishing Detection dataset from Kaggle ONCE and
    save it locally as CSV. If the CSV already exists, this does nothing
    (so we never hit Kaggle unnecessarily on future runs)."""
    if os.path.exists(config.CSV_PATH):
        print(f"{config.CSV_PATH} already exists — skipping download.")
        return

    import kagglehub
    import shutil
    print("Downloading dataset from Kaggle (first time only)...")
    path = kagglehub.dataset_download(config.KAGGLE_DATASET)

    # Find the CSV file inside the downloaded folder and copy it locally
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in downloaded Kaggle dataset.")
    shutil.copy(os.path.join(path, csv_files[0]), config.CSV_PATH)
    print(f"Saved dataset to {config.CSV_PATH}")


def load_dataset():
    """Load the dataset from the local CSV file (not from Kaggle every time)."""
    df = pd.read_csv(config.CSV_PATH)

    # Standardize column names so the rest of the pipeline (which expects
    # "URL" and "label" as 0/1) works the same way it did for PhiUSIIL.
    df = df.rename(columns={config.URL_COLUMN: "URL"})
    df["label"] = df[config.LABEL_COLUMN].map({"legitimate": 1, "phishing": 0})
    if config.LABEL_COLUMN != "label":
        df = df.drop(columns=[config.LABEL_COLUMN])

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

    Returns: X_train, X_val, y_train, y_val, train_idx, val_idx, split_details
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

    split_details = {
        "legit_train_idx": legit_train_idx, "legit_val_idx": legit_val_idx,
        "phish_train_idx": phish_train_idx, "phish_val_idx": phish_val_idx
    }

    return X_train, X_val, y_train, y_val, train_idx, val_idx, split_details


def check_no_leakage(train_idx, val_idx):
    """Sanity check: confirm no row appears in both train and validation sets."""
    overlap = train_idx.intersection(val_idx)
    print(f"Overlapping rows between train and validation: {len(overlap)}")
    if len(overlap) == 0:
        print("No data leakage detected.")
    else:
        print("WARNING: Data leakage detected!")
    return len(overlap) == 0


def exploratory_summary(df_numeric):
    """Print a full exploratory data analysis summary of the cleaned dataset:
    total rows, class balance, duplicate rows, and top feature correlations
    with the label. This documents the checks we ran to confirm the dataset
    is clean and that the model's high accuracy is legitimate (not a symptom
    of duplicated or leaking data)."""
    total_rows = len(df_numeric)
    class_counts = df_numeric["label"].value_counts()

    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total rows: {total_rows}")
    print(f"Legitimate (label=1): {class_counts.get(1, 0)} rows "
          f"({class_counts.get(1, 0)/total_rows*100:.1f}%)")
    print(f"Phishing   (label=0): {class_counts.get(0, 0)} rows "
          f"({class_counts.get(0, 0)/total_rows*100:.1f}%)")

    duplicate_count = df_numeric.duplicated().sum()
    print(f"\nExact duplicate rows: {duplicate_count} "
          f"({duplicate_count/total_rows*100:.2f}% of dataset)")

    correlations = df_numeric.corr()["label"].sort_values(ascending=False)
    print("\nTop 5 features most correlated with label:")
    print(correlations.head(6).drop("label", errors="ignore"))
    print("\nBottom 5 features (most negatively correlated):")
    print(correlations.tail(5))

    return {
        "total_rows": total_rows,
        "class_counts": class_counts,
        "duplicate_count": duplicate_count,
        "correlations": correlations
    }


def split_summary(legit_train_idx, legit_val_idx, phish_train_idx, phish_val_idx):
    """Print a breakdown confirming the stratified split was done fairly
    per class before merging into the final train/validation sets."""
    print("Stratified split breakdown (per class, before merge):")
    print(f"  Legitimate -> train: {len(legit_train_idx)}, validation: {len(legit_val_idx)}")
    print(f"  Phishing   -> train: {len(phish_train_idx)}, validation: {len(phish_val_idx)}")

    total_train = len(legit_train_idx) + len(phish_train_idx)
    total_val = len(legit_val_idx) + len(phish_val_idx)
    print(f"\nMerged TRAIN set size     : {total_train}")
    print(f"Merged VALIDATION set size : {total_val}")
