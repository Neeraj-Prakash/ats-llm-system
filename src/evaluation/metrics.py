import math


def precision_at_k(relevant, predicted, k):
    """
    Calculate the precision at k for a given set of relevant candidates and predicted candidates.

    Args:
        relevant (list): A list of relevant candidates to be used as ground truth.
        predicted (list): A list of predicted candidates returned by the algorithm.
        k (int): The cutoff value at which to measure precision.

    Returns:
        float: The precision at k for the given set of relevant and predicted candidates.
    """
    predicted_k = predicted[:k]
    relevant_set = set(relevant)
    hits = sum([1 for p in predicted_k if p in relevant_set])

    return hits / k


def recall_at_k(relevant, predicted, k):
    """
    Calculate the recall at k for a given set of relevant candidates and predicted candidates.

    Args:
        relevant (list): A list of relevant candidates to be used as ground truth.
        predicted (list): A list of predicted candidates returned by the algorithm.
        k (int): The cutoff value at which to measure recall.

    Returns:
        float: The recall at k for the given set of relevant and predicted candidates.
    """
    predicted_k = predicted[:k]

    relevant_set = set(relevant)
    hits = sum([1 for p in predicted_k if p in relevant_set])

    return hits / len(relevant_set) if relevant_set else 0


def dcg_at_k(relevances, k):
    """
    Calculate the Discounted Cumulative Gain (DCG) at k for a given list of relevances.

    Args:
        relevances (list): A list of relevance scores for each candidate in the predicted ranking.
        k (int): The cutoff value at which to measure DCG.

    Returns:
        float: The DCG at k for the given list of relevances and cutoff value.
    """
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += (2 ** relevances[i] - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(true_relevance_dict, predicted, k):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at k for a given set of true relevances and predicted candidates.

    Args:
        true_relevance_dict (dict): A dictionary mapping candidate ids to their true relevance scores.
        predicted (list): An ordered list of candidate ids returned by the algorithm.
        k (int): The cutoff value at which to measure NDCG.

    Returns:
        float: The NDCG at k for the given set of true relevances and predicted candidates.
    """

    # Get predicted relevance scores
    relevances = [true_relevance_dict.get(cid, 0) for cid in predicted]

    dcg = dcg_at_k(relevances, k)

    # Ideal ranking (sorted by true relevance score)
    ideal_relevances = sorted(true_relevance_dict.values(), reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0
