from __future__ import annotations

import csv
import re
import sqlite3
from datetime import datetime
from html import unescape
from pathlib import Path

import joblib
from flask import Flask, flash, redirect, render_template, request, url_for

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
INSTANCE_DIR = Path(__file__).resolve().parent / "instance"
DB_PATH = INSTANCE_DIR / "analysis_history.db"

MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

app = Flask(__name__)
app.config["SECRET_KEY"] = "local-phishing-email-detection-change-for-deployment"
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB request limit


def clean_text(text: str) -> str:
    """Match the cleaning rules used in src/preprocess.py."""
    text = unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def init_db() -> None:
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                sender TEXT NOT NULL,
                prediction TEXT NOT NULL,
                spam_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                checked_at TEXT NOT NULL
            )"""
        )
        conn.commit()


def get_history(limit: int = 8):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def get_stats():
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """SELECT COUNT(*),
               SUM(CASE WHEN prediction='Spam' THEN 1 ELSE 0 END),
               SUM(CASE WHEN prediction='Legitimate' THEN 1 ELSE 0 END)
               FROM analyses"""
        ).fetchone()
    return {"total": row[0] or 0, "spam": row[1] or 0, "legitimate": row[2] or 0}


def get_model_comparison():
    csv_path = RESULTS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


# Load the user's already-trained model and its matching TF-IDF vectorizer.
if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    raise FileNotFoundError(
        "Required model files are missing. Expected models/logistic_regression_model.pkl "
        "and models/tfidf_vectorizer.pkl."
    )
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
init_db()


@app.get("/")
def index():
    return render_template(
        "index.html",
        result=None,
        history=get_history(),
        stats=get_stats(),
        comparison=get_model_comparison(),
    )


@app.post("/analyze")
def analyze():
    subject = request.form.get("subject", "").strip()[:250]
    sender = request.form.get("sender", "").strip()[:250]
    body = request.form.get("body", "").strip()

    if not body and not subject:
        flash("Please enter an email subject or message before analysis.", "error")
        return redirect(url_for("index"))
    if len(body) > 50000:
        flash("Email text is too long. Please keep it under 50,000 characters.", "error")
        return redirect(url_for("index"))

    combined_text = clean_text(f"{subject} {body}")
    if not combined_text:
        flash("No readable email text was found. Please enter the subject or message.", "error")
        return redirect(url_for("index"))

    features = vectorizer.transform([combined_text])
    predicted_label = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    spam_index = classes.index(1) if 1 in classes else None
    spam_probability = float(probabilities[spam_index]) if spam_index is not None else 0.0

    prediction = "Spam" if predicted_label == 1 else "Legitimate"
    confidence = float(max(probabilities)) * 100
    if spam_probability >= 0.70:
        risk_level = "High"
    elif spam_probability >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    checked_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO analyses(subject, sender, prediction, spam_probability, risk_level, checked_at) VALUES (?, ?, ?, ?, ?, ?)",
            (subject or "(No subject)", sender or "(Not provided)", prediction,
             spam_probability * 100, risk_level, checked_at),
        )
        conn.commit()

    result = {
        "prediction": prediction,
        "spam_probability": spam_probability * 100,
        "confidence": confidence,
        "risk_level": risk_level,
        "subject": subject or "(No subject)",
        "sender": sender or "(Not provided)",
        "checked_at": checked_at,
    }
    return render_template(
        "index.html", result=result, history=get_history(),
        stats=get_stats(), comparison=get_model_comparison(),
    )


@app.errorhandler(413)
def request_too_large(_error):
    flash("The submitted request is too large. Please shorten the email text.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    print("\nPhishing Email Detection is starting...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
