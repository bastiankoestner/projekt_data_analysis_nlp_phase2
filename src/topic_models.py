from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from bertopic import BERTopic
from sklearn.decomposition import TruncatedSVD
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

@dataclass
class TopicResult:
    model_name: str
    topics_df: pd.DataFrame
    doc_topics: np.ndarray
    prevalence_df: pd.DataFrame

def _top_terms_per_topic(components, feature_names, top_n=12) -> pd.DataFrame:
    rows = []
    for topic_idx, weights in enumerate(components):
        top_idx = np.argsort(weights)[::-1][:top_n]
        terms = [feature_names[i] for i in top_idx]
        rows.append({"topic": topic_idx, "top_terms": ", ".join(terms)})
    return pd.DataFrame(rows)

def _prevalence_from_assignments(doc_topics: np.ndarray) -> pd.DataFrame:
    ser = pd.Series(doc_topics).value_counts().sort_values(ascending=False)
    df = ser.rename_axis("topic").reset_index(name="n_docs")
    df["share"] = df["n_docs"] / df["n_docs"].sum()
    return df

def run_nmf(X_tfidf, vectorizer, n_topics: int, random_state: int) -> TopicResult:
    nmf = NMF(
        n_components=n_topics,
        random_state=random_state,
        init="nndsvda",
        max_iter=300,
    )
    W = nmf.fit_transform(X_tfidf)
    doc_topics = W.argmax(axis=1)

    topics_df = _top_terms_per_topic(
        nmf.components_,
        vectorizer.get_feature_names_out(),
        top_n=12,
    )
    prevalence_df = _prevalence_from_assignments(doc_topics)

    return TopicResult("NMF", topics_df, doc_topics, prevalence_df)

def run_lda(X_tfidf, vectorizer, n_topics: int, random_state: int) -> TopicResult:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
        max_iter=20,
    )
    doc_topic = lda.fit_transform(X_tfidf)
    doc_topics = doc_topic.argmax(axis=1)

    topics_df = _top_terms_per_topic(
        lda.components_,
        vectorizer.get_feature_names_out(),
        top_n=12,
    )
    prevalence_df = _prevalence_from_assignments(doc_topics)

    return TopicResult("LDA", topics_df, doc_topics, prevalence_df)

def run_bertopic(texts: list[str], embeddings: np.ndarray, random_state: int) -> TopicResult:
    reducer = TruncatedSVD(n_components=10, random_state=random_state)
    hdbscan_model = HDBSCAN(min_cluster_size=15, core_dist_n_jobs=1)
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")

    topic_model = BERTopic(
        verbose=True,
        umap_model=reducer,              # <-- important: replaces UMAP
        hdbscan_model=hdbscan_model,     # <-- stable clustering
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False
    )

    topics, _ = topic_model.fit_transform(texts, embeddings)

    # Topics & Top Words
    info = topic_model.get_topic_info()
    rows = []
    for t in info["Topic"].tolist():
        if t == -1:
            continue
        words = topic_model.get_topic(t)
        top_terms = ", ".join([w for w, _ in words[:12]])
        rows.append({"topic": t, "top_terms": top_terms})
    topics_df = pd.DataFrame(rows).sort_values("topic")

    prevalence_df = (
        pd.Series(topics)
        .value_counts()
        .rename_axis("topic")
        .reset_index(name="n_docs")
        .sort_values("n_docs", ascending=False)
    )
    prevalence_df["share"] = prevalence_df["n_docs"] / prevalence_df["n_docs"].sum()

    return TopicResult("BERTopic", topics_df, np.array(topics), prevalence_df)