import numpy as np
import pandas as pd

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    relevances = np.asarray(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    gains = (2 ** relevances) - 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    ideal = dcg_at_k(np.sort(relevances)[::-1], k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal


def mean_ndcg(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    qid_col: str = "qid",
    k: int = 10,
) -> float:
    ndcg_scores = []

    for _, group in df.groupby(qid_col):
        ranked = group.sort_values(score_col, ascending=False)
        relevances = ranked[label_col].values
        ndcg_scores.append(ndcg_at_k(relevances, k))

    return float(np.mean(ndcg_scores))


def per_query_ndcg(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    qid_col: str = "qid",
    k: int = 10,
) -> pd.Series:
    results = {}
    for qid, group in df.groupby(qid_col):
        ranked = group.sort_values(score_col, ascending=False)
        results[qid] = ndcg_at_k(ranked[label_col].values, k)
    return pd.Series(results)
