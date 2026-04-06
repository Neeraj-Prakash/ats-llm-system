import json
import numpy as np
import os
import sys
import faiss

# Add the root directory (ats-llm-system) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Navigate to 'ats-llm-system' directory
sys.path.append(project_root)

from src.ranking.ranker import rerank_candidates
from src.utils.helpers import sort_candidates


CANDIDATES_PATH = "data/processed/candidates.json"
INDEX_PATH = "data/embeddings/faiss.index"
JOB_PATH = "data/processed/jobs.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    candidates = load_json(CANDIDATES_PATH)
    index = faiss.read_index(INDEX_PATH)
    jobs = load_json(JOB_PATH)
    job_id = 0

    job_description = jobs[job_id]["cleaned_text"]

    top_candidates, distances, ranking = rerank_candidates(
        job_description, candidates, index
    )

    results = sort_candidates(
        top_candidates, distances, ranking, sort_key="weighted_score"
    )

    print("\nTop 5 Candidates Based on Weighted Score:\n")
    for r in results[:5]:
        print(f"Candidate ID: {r['candidate_id']}")
        print(f"FAISS Score: {r['faiss_score']:.4f}")
        print(f"LLM Score: {r['llm_score']:.4f}")
        print(f"Final Score: {r['weighted_score']:.4f}")
        print("-" * 40)
