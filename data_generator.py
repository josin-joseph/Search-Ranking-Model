import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "bm25_body", "bm25_title", "bm25_url", "bm25_anchor",
    "tfidf_body", "tfidf_title", "tfidf_url", "tfidf_anchor",
    "lm_dir_body", "lm_dir_title", "lm_dir_url", "lm_dir_anchor",
    "overlap_body", "overlap_title", "overlap_url",
    "pagerank", "site_quality", "spam_score", "doc_length",
    "url_depth", "is_homepage",
    "ctr", "avg_dwell_time", "skip_rate", "click_entropy",
    "impressions", "clicks",
    "query_length", "query_term_idf_avg", "query_term_idf_max",
    "doc_age_days", "last_crawl_days",
    "title_coverage", "body_coverage", "anchor_coverage",
    "min_term_span", "avg_term_span",
    "bm25_x_pagerank", "tfidf_x_ctr", "overlap_x_quality",
    "query_doc_sim", "title_query_sim",
    "feat_42", "feat_43", "feat_44", "feat_45", "feat_46",
]


def generate_dataset(
    n_queries: int = 600,
    min_docs: int = 8,
    max_docs: int = 25,
    n_features: int = 46,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    records = []

    for qid in range(1, n_queries + 1):
        n_docs = np.random.randint(min_docs, max_docs + 1)
        labels = np.random.choice(
            [0, 1, 2, 3, 4],
            size=n_docs,
            p=[0.45, 0.28, 0.14, 0.09, 0.04],
        )

        for label in labels:
            rel = label / 4.0  # normalize relevance to [0, 1]

            bm25_body   = rel * 12 + np.random.normal(0, 2.5)
            bm25_title  = rel * 10 + np.random.normal(0, 2.0)
            bm25_url    = rel *  6 + np.random.normal(0, 1.5)
            bm25_anchor = rel *  7 + np.random.normal(0, 2.0)

            tfidf_body   = np.clip(rel * 0.80 + np.random.normal(0, 0.18), 0, 1)
            tfidf_title  = np.clip(rel * 0.85 + np.random.normal(0, 0.15), 0, 1)
            tfidf_url    = np.clip(rel * 0.55 + np.random.normal(0, 0.18), 0, 1)
            tfidf_anchor = np.clip(rel * 0.65 + np.random.normal(0, 0.20), 0, 1)

            lm_body   = rel * 11 + np.random.normal(0, 2.2)
            lm_title  = rel *  9 + np.random.normal(0, 2.0)
            lm_url    = rel *  5 + np.random.normal(0, 1.5)
            lm_anchor = rel *  7 + np.random.normal(0, 1.8)

            overlap_body  = np.clip(rel * 0.75 + np.random.normal(0, 0.16), 0, 1)
            overlap_title = np.clip(rel * 0.82 + np.random.normal(0, 0.14), 0, 1)
            overlap_url   = np.clip(rel * 0.50 + np.random.normal(0, 0.18), 0, 1)

            pagerank     = np.clip(rel * 0.60 + np.random.exponential(0.08), 0, 1)
            site_quality = np.clip(rel * 0.55 + np.random.exponential(0.10), 0, 1)
            spam_score   = np.clip((1 - rel) * 0.30 + np.random.exponential(0.05), 0, 1)
            doc_length   = np.random.exponential(600) + 100
            url_depth    = np.random.randint(1, 7)
            is_homepage  = float(np.random.binomial(1, 0.08 + rel * 0.05))

            ctr           = np.clip(rel * 0.32 + np.random.exponential(0.04), 0, 1)
            dwell_time    = rel * 180 + np.random.exponential(35)
            skip_rate     = np.clip((1 - rel) * 0.55 + np.random.exponential(0.08), 0, 1)
            click_entropy = np.clip(rel * 0.70 + np.random.normal(0, 0.15), 0, 1)
            impressions   = max(1, int(np.random.exponential(300) + 50))
            clicks        = min(impressions, max(0, int(rel * 80 + np.random.exponential(8))))

            query_length      = np.random.randint(1, 9)
            query_idf_avg     = np.random.uniform(3, 12)
            query_idf_max     = query_idf_avg + np.random.uniform(0, 5)

            doc_age       = np.random.exponential(400)
            last_crawl    = np.random.exponential(30)

            title_cov  = np.clip(rel * 0.80 + np.random.normal(0, 0.15), 0, 1)
            body_cov   = np.clip(rel * 0.75 + np.random.normal(0, 0.18), 0, 1)
            anchor_cov = np.clip(rel * 0.60 + np.random.normal(0, 0.20), 0, 1)

            min_span = max(1, int((1 - rel) * 50 + np.random.exponential(10)))
            avg_span = min_span + np.random.exponential(5)

            bm25_x_pr     = bm25_body * pagerank
            tfidf_x_ctr   = tfidf_body * ctr
            overlap_x_q   = overlap_body * site_quality
            query_doc_sim = np.clip(rel * 0.78 + np.random.normal(0, 0.14), 0, 1)
            title_q_sim   = np.clip(rel * 0.82 + np.random.normal(0, 0.13), 0, 1)

            extras = [
                rel * np.random.uniform(0.3, 0.7) + np.random.normal(0, 0.10)
                for _ in range(5)
            ]

            feature_values = [
                bm25_body, bm25_title, bm25_url, bm25_anchor,
                tfidf_body, tfidf_title, tfidf_url, tfidf_anchor,
                lm_body, lm_title, lm_url, lm_anchor,
                overlap_body, overlap_title, overlap_url,
                pagerank, site_quality, spam_score, doc_length,
                url_depth, is_homepage,
                ctr, dwell_time, skip_rate, click_entropy,
                impressions, clicks,
                query_length, query_idf_avg, query_idf_max,
                doc_age, last_crawl,
                title_cov, body_cov, anchor_cov,
                min_span, avg_span,
                bm25_x_pr, tfidf_x_ctr, overlap_x_q,
                query_doc_sim, title_q_sim,
            ] + extras

            records.append({
                "qid": qid,
                "label": label,
                **{f"f{i+1}": feature_values[i] for i in range(n_features)},
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Dataset shape : {df.shape}")
    print(f"Queries       : {df['qid'].nunique()}")
    print(f"Total docs    : {len(df)}")
    print(f"Label dist    :\n{df['label'].value_counts().sort_index()}")
