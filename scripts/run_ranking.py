import json
import numpy as np
import os
import sys
import faiss

# Add the root directory (ats-llm-system) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Navigate to 'ats-llm-system' directory
sys.path.append(project_root)

from src.embeddings.embedder import generate_embeddings


CANDIDATES_PATH = "data/processed/candidates.json"
INDEX_PATH = "data/embeddings/faiss.index"
JOB_PATH = "data/raw/jobs/em_01.txt"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def read_job_description(path):
    with open(path, "r") as f:
        return f.read().strip()


if __name__ == "__main__":
    candidates = load_json(CANDIDATES_PATH)

    index = faiss.read_index(INDEX_PATH)

    job_description = read_job_description(JOB_PATH)

    query_embedding = generate_embeddings(job_description, is_query=True)
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k=5)

    print("\nTop Matches:\n")

    for i, idx in enumerate(indices[0]):
        print(f"Rank {i+1}")
        print(f"Candidate ID: {candidates[idx]['candidate_id']}")
        print(f"Skills: {candidates[idx]['skills']}")
        print(f"Distance: {distances[0][i]}")
        print("-" * 40)
