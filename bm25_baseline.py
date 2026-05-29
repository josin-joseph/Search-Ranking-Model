import pandas as pd

BM25_FEATURE = "f1"  

def rank_by_bm25(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bm25_score"] = df[BM25_FEATURE]
    return df
