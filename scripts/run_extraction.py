import os
import json
import sys

# Add the root directory (ats-llm-system) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Navigate to 'ats-llm-system' directory
sys.path.append(project_root)

# Now import the function
from src.ingestion.pdf_loader import extract_all_resumes
from src.ingestion.text_cleaner import clean_all_resumes

from src.extraction.llm_extractor import summarize_all_resumes
from src.utils.model_loader import get_model


def save_to_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # Path to your dataset
    ROOT_DIR = "data/raw/resumes/data"

    # Output file
    OUTPUT_PATH = "data/interim/extracted_text/resumes_text.json"

    # Model
    model, tokenizer = get_model(seq_length=5120)

    print("Starting PDF extraction pipeline...")

    dataset = extract_all_resumes(ROOT_DIR)
    print(f"Extracted {len(dataset)} resumes from PDFs.")

    print("Starting text cleaning pipeline...")
    cleaned_dataset = clean_all_resumes(dataset)
    print(f"Cleaned {len(cleaned_dataset)} resumes from PDFs.")

    print("Starting text summarization pipeline with LLM...")
    summarized_dataset = summarize_all_resumes(cleaned_dataset, model, tokenizer)
    print(f"Summarized {len(summarized_dataset)} resumes from PDFs.")

    print(f"Saving {len(summarized_dataset)} records...")

    save_to_json(summarized_dataset, OUTPUT_PATH)

    print("✅ Extraction completed successfully!")
