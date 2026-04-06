import os, sys
import json
import faiss

# Add the root directory (ats-llm-system) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.evaluation.evaluator import evaluate_ranking
from src.ranking.ranker import rerank_candidates
from src.utils.helpers import sort_candidates
from src.utils.model_loader import get_model

CANDIDATES_PATH = "data/processed/candidates.json"
INDEX_PATH = "data/embeddings/faiss.index"
JOB_PATH = "data/processed/jobs.json"
LABELS_PATH = "data/processed/labels.json"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    candidates = load_json(CANDIDATES_PATH)
    index = faiss.read_index(INDEX_PATH)
    jobs = load_json(JOB_PATH)
    labels_data = load_json(LABELS_PATH)
    model, tokenizer = get_model(seq_length=5120)

    all_results = []

    for job_entry in jobs:
        job_description = job_entry["cleaned_text"]
        job_id = job_entry["job_id"]
        true_labels = labels_data[job_id]["labels"]

        top_candidates, distances, ranking = rerank_candidates(
            job_description, candidates, index, model, tokenizer
        )
        predicted = sort_candidates(
            top_candidates, distances, ranking, sort_key="llm_score"
        )
        predicted_ids = [tc["candidate_id"] for tc in predicted]
        results = evaluate_ranking(true_labels, predicted_ids, k=5)

        print(f"\nJob: {job_id}")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

        all_results.append(results)

    # Aggregate results
    avg_precision = sum(r["precision@k"] for r in all_results) / len(all_results)
    avg_recall = sum(r["recall@k"] for r in all_results) / len(all_results)
    avg_ndcg = sum(r["ndcg@k"] for r in all_results) / len(all_results)

    print("\n=== Overall Performance ===")
    print(f"Avg Precision@K: {avg_precision:.4f}")
    print(f"Avg Recall@K: {avg_recall:.4f}")
    print(f"Avg NDCG@K: {avg_ndcg:.4f}")
