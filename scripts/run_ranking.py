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


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    candidates = load_json(CANDIDATES_PATH)

    index = faiss.read_index(INDEX_PATH)

    job_description = """
    FOX Corporation is looking for an experienced Engineering Manager to lead our web engineering team responsible for building high-quality, scalable, and reliable web applications. You will be responsible for building, mentoring, and scaling a high-performing team while driving technical excellence and delivery of world-class web experiences.
    """

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
