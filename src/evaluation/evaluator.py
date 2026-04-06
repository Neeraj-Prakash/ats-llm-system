from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k


def evaluate_ranking(true_labels, predicted, k=5):
    """
    Evaluate the ranking of a model using the true labels and predicted scores for k items.

    Args:
        true_labels (dict): A dictionary containing the true labels for each item,
        predicted (dict): A dictionary containing the predicted scores for each item
        k (int): The number of top items to consider for evaluation. Default is 5.

    Returns:
        dict: A dictionary containing the precision@k, recall@k, and ndcg@k metrics for the model.
    """

    relevant = [cid for cid, score in true_labels.items() if score > 3]

    precision = precision_at_k(relevant, predicted, k)
    recall = recall_at_k(relevant, predicted, k)
    ndcg = ndcg_at_k(true_labels, predicted, k)

    return {"precision@k": precision, "recall@k": recall, "ndcg@k": ndcg}
