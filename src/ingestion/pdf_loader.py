import os
import pdfplumber
from typing import List, Dict
from tqdm import tqdm


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a single PDF file.
    """
    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")

    return text.strip()


def get_all_pdf_paths(root_dir: str) -> List[str]:
    """
    Recursively collect all PDF file paths from nested directories.
    """
    pdf_paths = []
    # Ensure the root directory exists
    if not os.path.isdir(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist or is not a directory")
        return pdf_paths

    for root, dirs, files in os.walk(root_dir):
        print(root_dir)
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))

    print(f"Found {len(pdf_paths)} PDF files in directories:")

    return pdf_paths


def extract_all_resumes(root_dir: str) -> List[Dict]:
    """
    Extract text from all PDFs and return structured list.
    """
    pdf_paths = get_all_pdf_paths(root_dir)

    if not pdf_paths:
        print("No PDF files found. Check directory permissions and file extensions.")
        return []

    dataset = []

    print(f"\nProcessing {len(pdf_paths)} PDF files...")

    for idx, pdf_path in tqdm(enumerate(pdf_paths), total=len(pdf_paths)):
        try:
            # Check if file exists before processing
            if not os.path.exists(pdf_path):
                print(f"  Skipping: File not found at {pdf_path}")
                continue

            text = extract_text_from_pdf(pdf_path)

            candidate = {
                "candidate_id": f"C{idx+1:04d}",
                "file_name": os.path.basename(pdf_path),
                "file_path": pdf_path,
                "resume_text": text,
            }

            dataset.append(candidate)

        except Exception as e:
            print(f"  Error processing {pdf_path}: {str(e)}")

    return dataset


if __name__ == "__main__":
    # Update this path to your actual directory
    root_directory = r"ats-llm-system\data\raw\resumes\data"

    resumes = extract_all_resumes(root_directory)

    print(f"\nSuccessfully processed {len(resumes)} resumes.")
