def sort_candidates(top_candidates, distances, ranking, sort_key="weighted_score"):
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
    results.sort(key=lambda x: x[sort_key], reverse=True)
    return results
