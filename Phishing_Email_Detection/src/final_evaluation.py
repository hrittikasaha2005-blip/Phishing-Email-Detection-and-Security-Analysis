
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

print("Loading testing data...")

X_test = load_npz(DATA_DIR / "X_test.npz")
y_test = np.load(DATA_DIR / "y_test.npy")

# Load selected model
model = joblib.load(
    MODEL_DIR / "logistic_regression_model.pkl"
)

print("Logistic Regression model loaded!")

# Generate predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nFINAL MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Spam"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print(cm)

# Save classification report
report_path = RESULTS_DIR / "final_classification_report.txt"

with open(report_path, "w") as file:
    file.write("FINAL MODEL: LOGISTIC REGRESSION\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))

# Plot confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Spam"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

graph_path = RESULTS_DIR / "final_confusion_matrix.png"

plt.savefig(graph_path, dpi=300)
plt.show()

print("\nEvaluation report saved:", report_path)
print("Confusion matrix saved:", graph_path)

print("\nSTEP 23 COMPLETED SUCCESSFULLY!")
