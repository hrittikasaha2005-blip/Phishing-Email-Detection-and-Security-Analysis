
import csv
import re
from pathlib import Path
from email import policy
from email.parser import BytesParser
from html import unescape
from html.parser import HTMLParser


# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

# Output CSV file
OUTPUT_FILE = DATA_DIR / "emails.csv"


class HTMLTextExtractor(HTMLParser):
    """Extract readable text from HTML emails."""

    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts)


def clean_text(text):
    """Clean email text for machine learning."""

    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_email_body(message):
    """Extract plain text from an email."""

    plain_parts = []
    html_parts = []

    parts = message.walk() if message.is_multipart() else [message]

    for part in parts:

        if part.get_content_disposition() == "attachment":
            continue

        content_type = part.get_content_type()

        if content_type not in ["text/plain", "text/html"]:
            continue

        try:
            content = part.get_content()
        except Exception:
            continue

        if content_type == "text/plain":
            plain_parts.append(str(content))

        elif content_type == "text/html":
            html_parts.append(str(content))

    if plain_parts:
        return " ".join(plain_parts)

    if html_parts:
        extractor = HTMLTextExtractor()
        extractor.feed(" ".join(html_parts))
        return extractor.get_text()

    return ""


def process_dataset(folder_name, category, label):

    folder = DATA_DIR / folder_name
    records = []

    for file_path in folder.iterdir():

        # Skip README and other non-email files
        if not file_path.is_file():
            continue

        if not re.match(r"^\d{5}\.", file_path.name):
            continue

        try:
            with open(file_path, "rb") as file:
                message = BytesParser(policy=policy.default).parse(file)

            subject = str(message.get("subject", ""))
            sender = str(message.get("from", ""))

            body = get_email_body(message)

            cleaned_subject = clean_text(subject)
            cleaned_body = clean_text(body)

            full_text = cleaned_subject + " " + cleaned_body

            records.append({
                "label": label,
                "category": category,
                "subject": subject,
                "sender": sender,
                "body": body,
                "cleaned_text": full_text,
                "file_name": file_path.name
            })

        except Exception as error:
            print(f"Error processing {file_path.name}: {error}")

    print(f"{category}: {len(records)} emails processed")

    return records


def main():

    print("Starting email preprocessing...\n")

    ham_records = process_dataset("easy_ham", "Ham", 0)
    spam_records = process_dataset("spam", "Spam", 1)

    all_records = ham_records + spam_records

    fieldnames = [
        "label",
        "category",
        "subject",
        "sender",
        "body",
        "cleaned_text",
        "file_name"
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(all_records)

    print("\nPreprocessing completed successfully!")
    print(f"Total emails processed: {len(all_records)}")
    print(f"Legitimate emails: {len(ham_records)}")
    print(f"Spam emails: {len(spam_records)}")
    print(f"\nDataset saved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
