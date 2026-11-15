# PhishGuard — Phishing Email Detection

A local Flask web application that uses the trained Logistic Regression model and its matching TF-IDF vectorizer from this project. The application also keeps a small local SQLite history (subject, sender, result, score and timestamp); it does not store the email body.

## Quick start (Windows)

1. Extract the complete `Phishing_Email_Detection` folder.
2. Make sure the `models` folder contains `logistic_regression_model.pkl` and `tfidf_vectorizer.pkl`.
3. Double-click `run.bat`.
4. On first run, the script creates `.venv` and installs packages. Internet access is required for this first installation.
5. When Flask starts, open **http://127.0.0.1:5000** in your browser.
6. To stop the server, return to the command window and press `Ctrl+C`.

You can also run manually from PowerShell in the project folder:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app\app.py
```

If PowerShell blocks activation, skip activation and use `.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `.venv\Scripts\python.exe app\app.py`.

## What is included

- `app/`: Flask application, dashboard template, stylesheet and local SQLite database folder.
- `models/`: the three trained classifiers and the TF-IDF vectorizer supplied in the uploaded project.
- `src/`: original preprocessing, feature preparation, training, comparison and evaluation scripts.
- `data/`: supplied email dataset and processed train/test matrices.
- `results/`: supplied model comparison and evaluation outputs.
- `run.bat`: Windows setup and launch helper.

## Model and preprocessing notes

The app uses `logistic_regression_model.pkl` with `tfidf_vectorizer.pkl`. It combines the subject and body (the sender is retained for history but is not fed into the model, matching the original training pipeline) and applies the same text-cleaning rules as `src/preprocess.py`: HTML-like tags are removed, URLs are replaced with the token `URL`, HTML entities are unescaped, and whitespace is normalized. The app then calls the saved vectorizer's `transform` method; it does not fit a new vectorizer.

The spam probability shown is the model's `predict_proba` output for class `1`. The Low/Medium/High risk bands are display thresholds (below 40%, 40–69.9%, and 70% or above), not independently validated security grades. Model accuracy shown in the dashboard (99.17%) is the reported held-out test accuracy from the supplied evaluation; it is not a guarantee for new emails.

## Re-running the original ML workflow

Use the same Python environment and the supplied `requirements.txt`. The original scripts are in `src/`. `src/preprocess.py` expects the extracted `data/easy_ham` and `data/spam` folders. Feature preparation and model scripts use `data/processed/` and `models/` as in the original source.

## Important

This is an academic screening aid. A Legitimate result does not guarantee an email is safe, and a Spam result is not definitive proof of malicious intent. Do not open suspicious links or disclose credentials based only on this result. The Flask server binds to `127.0.0.1` so it is accessible only from the local computer by default.
