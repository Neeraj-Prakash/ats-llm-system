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
from src.ranking.ranker import rerank_candidates
from src.extraction.llm_extractor import model, tokenizer


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

    distances, indices = index.search(query_embedding, k=10)
    top_candidates = [candidates[i] for i in indices[0]]

    reranked = rerank_candidates(model, tokenizer, job_description, top_candidates)

    ranking = reranked.get("ranking", [])

    results = []
    for r in ranking:
        idx = r["candidate_index"] - 1
        candidate = top_candidates[idx]
        faiss_score = distances[0][idx]
        normalized_faiss = (faiss_score + 1) / 2
        weighted_score = 0.7 * normalized_faiss + 0.3 * (r["score"] / 10)
        results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "faiss_score": faiss_score,
                "llm_score": r["score"],
                "weighted_score": weighted_score,
            }
        )
    results.sort(key=lambda x: x["weighted_score"], reverse=True)

    print("\nTop 5 Candidates Based on Weighted Score:\n")
    for r in results[:5]:
        print(f"Candidate ID: {r['candidate_id']}")
        print(f"FAISS Score: {r['faiss_score']:.4f}")
        print(f"LLM Score: {r['llm_score']:.4f}")
        print(f"Final Score: {r['weighted_score']:.4f}")
        print("-" * 40)
