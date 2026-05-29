import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit


FEATURE_COLS = [f"f{i}" for i in range(1, 47)]  # f1 to f46


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
):
    qids = df["qid"].values
    unique_qids = df["qid"].unique()

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=qids))

    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def train_lambdamart(
    train_df: pd.DataFrame,
    n_estimators: int = 200,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> lgb.LGBMRanker:
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values

    group_sizes = train_df.groupby("qid").size().values

    model = lgb.LGBMRanker(
        objective="lambdarank",   
        metric="ndcg",
        ndcg_eval_at=[10],        
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        group=group_sizes,
    )

    return model


def predict_scores(model: lgb.LGBMRanker, df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    X = df[FEATURE_COLS].values
    df["lambdamart_score"] = model.predict(X)
    return df


def get_feature_importance(model: lgb.LGBMRanker, top_n: int = 15) -> pd.DataFrame:
    """
    Returns top N most important features by gain.
    """
    from data.data_generator import FEATURE_NAMES

    n = len(model.feature_importances_)
    names = FEATURE_NAMES[:n] if n <= len(FEATURE_NAMES) else [f"f{i+1}" for i in range(n)]

    importance = pd.DataFrame({
        "feature": names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return importance.head(top_n)
