
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import load_npz

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading TF-IDF features...")

# Load training and testing datasets
X_train = load_npz(DATA_DIR / "X_train.npz")
X_test = load_npz(DATA_DIR / "X_test.npz")

y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Initialize Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
)

print("\nTraining Random Forest model...")

# Train model
model.fit(X_train, y_train)

print("Training completed!")

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("-------------------------")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Spam"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save trained model
model_path = MODEL_DIR / "random_forest_model.pkl"

joblib.dump(model, model_path)

print("\nRandom Forest model saved successfully!")
print("Model location:", model_path)
