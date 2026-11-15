
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

print("Loading testing data...")

# Load the same testing dataset
X_test = load_npz(DATA_DIR / "X_test.npz")
y_test = np.load(DATA_DIR / "y_test.npy")

print("Testing data:", X_test.shape)

# Load all three trained models
models = {
    "Naive Bayes": joblib.load(
        MODEL_DIR / "naive_bayes_model.pkl"
    ),

    "Logistic Regression": joblib.load(
        MODEL_DIR / "logistic_regression_model.pkl"
    ),

    "Random Forest": joblib.load(
        MODEL_DIR / "random_forest_model.pkl"
    )
}

results = []

print("\nMODEL COMPARISON")
print("=" * 50)

# Evaluate each model
for name, model in models.items():

    print(f"\nEvaluating {name}...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1-Score": f1 * 100
    })

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")

# Create comparison dataframe
df = pd.DataFrame(results)

print("\nFINAL MODEL COMPARISON")
print("=" * 70)
print(df.round(2).to_string(index=False))

# Save comparison results
csv_path = RESULTS_DIR / "model_comparison.csv"
df.to_csv(csv_path, index=False)

print("\nComparison CSV saved:", csv_path)

# Plot performance comparison
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

ax = df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6),
    width=0.75
)

ax.set_title("Performance Comparison of ML Models")
ax.set_ylabel("Score (%)")
ax.set_xlabel("Machine Learning Models")
ax.set_ylim(0, 105)

plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

graph_path = RESULTS_DIR / "model_comparison.png"
plt.savefig(graph_path, dpi=300)

plt.show()

print("\nComparison graph saved:", graph_path)

# Identify the best model by accuracy
best_model = df.loc[df["Accuracy"].idxmax()]

print("\nBEST PERFORMING MODEL")
print("=" * 50)
print("Model:", best_model["Model"])
print(f"Accuracy: {best_model['Accuracy']:.2f}%")

print("\nSTEP 22 COMPLETED SUCCESSFULLY!")
