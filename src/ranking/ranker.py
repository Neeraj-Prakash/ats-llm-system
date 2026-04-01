import torch
import os
import sys

# Add the root directory (ats-llm-system) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Navigate to 'ats-llm-system' directory
sys.path.append(project_root)
from src.embeddings.embedder import build_candidate_text

def build_rerank_prompt(job_desc, candidates):
    prompt = f"""
You are an expert recruiter.

Given a job description and a list of candidates, rank the candidates from best to worst.

Job Description:
{job_desc}

Candidates:
"""

    for i, c in enumerate(candidates):
        prompt += f"""
Candidate {i+1}:

"""
        prompt += build_candidate_text(c)

    prompt += """
Return ONLY JSON in this format:
{
  "ranking": [
    {"candidate_index": 1, "score": 9.5},
    {"candidate_index": 2, "score": 5.1}
  ]
}
Rules:
- Limit the number of candidates to 10.
- The score should be between 0 and 10.
"""

    return prompt


def safe_json_extract(text: str):
    import re, json

    text = text.replace("```json", "").replace("```", "")

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return {"ranking": []}


def rerank_candidates(model, tokenizer, job_desc, candidates):
    prompt = build_rerank_prompt(job_desc, candidates)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=5120
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=600, temperature=0.1, do_sample=False
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    decoded = decoded[len(prompt) :]
    
    result = safe_json_extract(decoded)

    return result
