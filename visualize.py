import matplotlib.pyplot as plt
import numpy as np
from evaluate import mean_ndcg

def plot_ndcg_comparison(bm25_score, lm_score):
    models = ["BM25", "LambdaMART"]
    scores = [bm25_score, lm_score]
    colors = ["#9b9b9b", "#2563eb"]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(models, scores, color=colors, width=0.4)

    # value on top of each bar
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f"{score:.4f}", ha="center", fontsize=12)

    plt.title("BM25 vs LambdaMART — NDCG@10")
    plt.ylabel("NDCG@10")
    plt.ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig("outputs/ndcg_comparison.png", dpi=150)
    plt.close()
    print("Saved → outputs/ndcg_comparison.png")


def plot_ndcg_curve(test_df):
    k_values = list(range(1, 21))
    bm25_scores = [mean_ndcg(test_df, "bm25_score", k=k) for k in k_values]
    lm_scores   = [mean_ndcg(test_df, "lambdamart_score", k=k) for k in k_values]

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, bm25_scores, "o-", color="#9b9b9b", label="BM25")
    plt.plot(k_values, lm_scores,   "o-", color="#2563eb", label="LambdaMART")
    plt.fill_between(k_values, bm25_scores, lm_scores, alpha=0.1, color="#2563eb")

    plt.title("NDCG@K Curve — BM25 vs LambdaMART")
    plt.xlabel("K")
    plt.ylabel("NDCG@K")
    plt.xticks(k_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/ndcg_curve.png", dpi=150)
    plt.close()
    print("Saved → outputs/ndcg_curve.png")