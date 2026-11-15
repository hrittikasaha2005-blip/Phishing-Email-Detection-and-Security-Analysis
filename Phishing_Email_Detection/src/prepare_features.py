
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.sparse import save_npz

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_DIR / "data" / "emails.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading email dataset...")

    df = pd.read_csv(DATA_FILE)

    # Validate required columns
    if "cleaned_text" not in df.columns or "label" not in df.columns:
        raise ValueError("Required columns are missing from emails.csv")

    # Prepare text and labels
    X = df["cleaned_text"].fillna("").astype(str)
    y = pd.to_numeric(df["label"], errors="coerce")

    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    y = y.astype(int)

    print("Total emails:", len(df))

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nDataset splitting completed!")
    print("Training emails:", len(X_train))
    print("Testing emails:", len(X_test))

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True
    )

    print("\nApplying TF-IDF...")

    # Fit only on training text to prevent data leakage
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform testing text using the same vocabulary
    X_test_tfidf = vectorizer.transform(X_test)

    # Save TF-IDF matrices
    save_npz(
        PROCESSED_DIR / "X_train.npz",
        X_train_tfidf
    )

    save_npz(
        PROCESSED_DIR / "X_test.npz",
        X_test_tfidf
    )

    # Save corresponding labels
    np.save(
        PROCESSED_DIR / "y_train.npy",
        y_train.to_numpy()
    )

    np.save(
        PROCESSED_DIR / "y_test.npy",
        y_test.to_numpy()
    )

    # Save vectorizer for future predictions
    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    print("\nTF-IDF feature extraction completed!")

    print("\nFeature matrix information:")
    print("Training matrix:", X_train_tfidf.shape)
    print("Testing matrix:", X_test_tfidf.shape)
    print("Vocabulary size:", len(vectorizer.vocabulary_))

    print("\nTraining label distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting label distribution:")
    print(y_test.value_counts().sort_index())

    print("\nAll feature files saved successfully!")


if __name__ == "__main__":
    main()
