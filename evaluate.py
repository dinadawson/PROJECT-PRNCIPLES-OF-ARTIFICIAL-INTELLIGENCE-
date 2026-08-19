"""
evaluate.py
Functions to evaluate the trained model and visualize / export results:
- accuracy, precision, recall, specificity, F1
- confusion matrix (image)
- feature importance (image)
- full prediction tables (Ground Truth, Predicted, Score) exported to Excel
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import config


def evaluate_model(model, X_val, y_val):
    """Run predictions on the validation set and print accuracy + classification report."""
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    return y_pred, accuracy


def plot_confusion_matrix(y_val, y_pred):
    """Generate and save the confusion matrix image, and print the explicit
    TP/TN/FP/FN breakdown with all derived metrics."""
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Phishing", "Legitimate"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Phishing Detection (Random Forest)")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"TP: {tp}   TN: {tn}   FP: {fp}   FN: {fn}")
    print(f"Accuracy: {acc:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  "
          f"Specificity: {specificity:.4f}  F1: {f1:.4f}")

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1}


def plot_feature_importance(model, X):
    """Generate and save a bar chart of the top 15 most important features."""
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, hue=top_features.index,
                palette="viridis", legend=False)
    plt.title("Top 15 Most Important Features - Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(config.FEATURE_IMPORTANCE_PATH)
    plt.show()

    return top_features


def export_prediction_table(model, X, y, url_text, output_path):
    """Generate a full prediction table (Ground Truth, Predicted, Score) for
    every row in X, and export it to Excel. Vectorized (not manual/looped)."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    scores = probabilities.max(axis=1)

    results_df = pd.DataFrame({
        "Index": X.index,
        "URL": url_text.loc[X.index].values if url_text is not None else "N/A",
        "Ground_Truth": y.values,
        "Predicted": predictions,
        "Score": scores.round(4)
    })
    results_df["Ground_Truth_Label"] = results_df["Ground_Truth"].map({0: "Phishing", 1: "Legitimate"})
    results_df["Predicted_Label"] = results_df["Predicted"].map({0: "Phishing", 1: "Legitimate"})
    results_df["Correct"] = results_df["Ground_Truth"] == results_df["Predicted"]

    results_df.to_excel(output_path, index=False)
    print(f"Saved {output_path} ({len(results_df)} rows)")
    print(f"Accuracy check: {results_df['Correct'].mean():.4f}")

    return results_df
