import re
import json
from tqdm import tqdm

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer


MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

max_seq_length = 2048
dtype = None  # auto
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)
print("✅ Model loaded successfully!")


def build_prompt(resume_text: str) -> str:
    return f"""
You are an expert resume parser.

Extract the following fields from the resume:
- skills (list of strings)
- total_experience (years - calculate from past roles, number)
- education (list containing {'degree', 'university'})
- current_role (string)
- past_roles (list containing {'role', 'summary'})

Ensure that the extracted data is in valid JSON format. If any data is missing or incorrect, return it as an empty array or string.
Rules:
- Limit the number of skills to a maximum of 20.
- Limit the number of past roles to a maximum of 5.
- Keep past role summaries concise.

Resume:
{resume_text}

JSON Output:
```json
"""


def get_empty_schema():
    """Return an empty schema for validation"""
    return {
        "skills": [],
        "total_experience": 0,
        "education": [],
        "current_role": "",
        "past_roles": [],
    }


def safe_json_extract(text: str):
    """
    Extract JSON block from model output
    """
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            raise ValueError("No JSON found")

    except Exception as e:
        print(f"[WARNING] JSON parsing failed: {e}")
        return get_empty_schema()


def extract_with_llm(resume_text: str):
    """
    Extract data from the model's response using a prompt.
    """

    prompt = build_prompt(resume_text)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1000, temperature=0.5, do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded = decoded[len(prompt) :]

    return safe_json_extract(decoded)


def validate_extracted_data(data: dict[str, any]) -> dict[str, any]:
    """
    Ensure schema consistency
    """

    if not isinstance(data.get("skills"), list):
        data["skills"] = []

    if not isinstance(data.get("education"), list):
        data["education"] = []

    if not isinstance(data.get("past_roles"), list):
        data["past_roles"] = []

    if not isinstance(data.get("total_experience"), (int, float)):
        data["total_experience"] = 0

    return data


def summarize_all_resumes(resume_data: list[dict[str, any]]) -> dict[str, any]:
    """
    Summarize all resumes
    """
    summarized_data = []
    for resume in tqdm(resume_data):
        structured = extract_with_llm(resume["cleaned_text"])
        structured = validate_extracted_data(structured)
        resume.update(structured)
        summarized_data.append(resume)
    return summarized_data
