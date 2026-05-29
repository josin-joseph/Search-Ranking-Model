import os
os.makedirs("outputs", exist_ok=True)

from data_generator import generate_dataset
from bm25_baseline import rank_by_bm25
from lambdamart_model import get_train_test_split, train_lambdamart, predict_scores
from evaluate import mean_ndcg
from visualize import plot_ndcg_comparison, plot_ndcg_curve

# data
df = generate_dataset()
train_df, test_df = get_train_test_split(df)

# train
model = train_lambdamart(train_df)

# score
test_df = rank_by_bm25(test_df)
test_df = predict_scores(model, test_df)

# evaluate
bm25_score = mean_ndcg(test_df, score_col="bm25_score", k=10)
lm_score   = mean_ndcg(test_df, score_col="lambdamart_score", k=10)

print(f"BM25       NDCG@10: {bm25_score:.4f}")
print(f"LambdaMART NDCG@10: {lm_score:.4f}")
print(f"Improvement: +{(lm_score - bm25_score) / bm25_score * 100:.1f}%")

# plots
plot_ndcg_comparison(bm25_score, lm_score)
plot_ndcg_curve(test_df)