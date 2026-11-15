
import pandas as pd
from pathlib import Path

# Locate project dataset
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "emails.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "emails.csv"

print("Loading email dataset...")

df = pd.read_csv(INPUT_FILE)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill missing text fields without deleting emails
text_columns = [
    "subject",
    "sender",
    "body",
    "cleaned_text"
]

for column in text_columns:
    df[column] = df[column].fillna("")

# Remove unnecessary leading/trailing whitespace
for column in text_columns:
    df[column] = df[column].astype(str).str.strip()

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset cleaning completed!")
print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("Saved to:", OUTPUT_FILE)
