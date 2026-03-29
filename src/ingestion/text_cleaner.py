import re
from tqdm import tqdm

def basic_clean(text: str) -> str:
    """
    Basic cleaning:
    - Normalize whitespace
    - Remove weird characters
    """
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove non-ASCII (optional, keep if needed)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    return text.strip()


def remove_urls_emails(text: str) -> str:
    """
    Remove emails and URLs (optional depending on use-case)
    """
    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    return text


def normalize_text(text: str) -> str:
    """
    Normalize text:
    - Lowercase
    - Standardize spacing
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_extra_symbols(text: str) -> str:
    """
    Remove unnecessary symbols but keep useful ones
    """
    text = re.sub(r"[^\w\s.,+#/-]", " ", text)
    return text


def clean_resume_text(text: str) -> str:
    """
    Full cleaning pipeline
    """
    text = basic_clean(text)
    text = remove_urls_emails(text)
    text = remove_extra_symbols(text)
    text = normalize_text(text)

    return text


def clean_all_resumes(resume_data: list[dict[str, any]]) -> list[dict[str, any]]:
    # Iterate over each resume and apply the cleaning function
    cleaned_data = []
    for resume in tqdm(resume_data):
        cleaned_resume = clean_resume_text(resume["resume_text"])
        resume["cleaned_text"] = cleaned_resume
        cleaned_data.append(resume)

    return cleaned_data
