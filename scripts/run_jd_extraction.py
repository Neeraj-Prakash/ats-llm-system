import os
import json
import sys

# Add the root directory (ats-llm-system) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Navigate to 'ats-llm-system' directory
sys.path.append(project_root)

# Now import the function
from src.ingestion.text_loader import extract_all_jobs
from src.ingestion.text_cleaner import clean_all_jds


def save_to_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Path to your dataset
    ROOT_DIR = "data/raw/jobs"

    # Output file
    OUTPUT_PATH = "data/processed/jobs.json"

    print("Starting PDF extraction pipeline...")

    dataset = extract_all_jobs(ROOT_DIR)
    print(f"Extracted {len(dataset)} resumes from PDFs.")

    print("Starting text cleaning pipeline...")
    cleaned_dataset = clean_all_jds(dataset)
    print(f"Cleaned {len(cleaned_dataset)} resumes from PDFs.")

    print(f"Saving {len(cleaned_dataset)} records...")

    save_to_json(cleaned_dataset, OUTPUT_PATH)

    print("✅ Extraction completed successfully!")
