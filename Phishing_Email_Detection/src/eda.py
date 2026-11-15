
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "emails.csv"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

print("DATASET OVERVIEW")
print("-------------------------")
print("Total emails:", len(df))
print("\nCategory distribution:")
print(df["category"].value_counts())

# 1. Ham vs Spam distribution
counts = df["category"].value_counts()

plt.figure(figsize=(8, 5))
counts.plot(kind="bar")

plt.title("Ham vs Spam Email Distribution")
plt.xlabel("Email Category")
plt.ylabel("Number of Emails")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "email_distribution.png")
plt.close()

# 2. Email length analysis
df["email_length"] = df["cleaned_text"].fillna("").str.len()

plt.figure(figsize=(8, 5))

df.boxplot(
    column="email_length",
    by="category"
)

plt.title("Email Length by Category")
plt.suptitle("")
plt.xlabel("Email Category")
plt.ylabel("Number of Characters")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "email_length.png")
plt.close()

print("\nEDA completed successfully!")
print("Graphs saved in the results folder.")
