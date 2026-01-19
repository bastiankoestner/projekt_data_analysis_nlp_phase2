from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from bertopic import BERTopic
from sklearn.decomposition import TruncatedSVD
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


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
        umap_model=reducer,
        hdbscan_model=hdbscan_model,
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
    prevalence_df = prevalence_df[prevalence_df["topic"] != -1]

    prevalence_df["share"] = prevalence_df["n_docs"] / prevalence_df["n_docs"].sum()

    return TopicResult("BERTopic", topics_df, np.array(topics), prevalence_df)

def _components_to_wordlists(components, feature_names, top_n: int = 12) -> list[list[str]]:
    topic_wordlists = []
    for weights in components:
        top_idx = np.argsort(weights)[::-1][:top_n]
        topic_wordlists.append([feature_names[i] for i in top_idx])
    return topic_wordlists

def compute_coherence(
    topic_wordlists: list[list[str]],
    tokenized_texts: list[list[str]],
    coherence: str = "c_v",
) -> float:
    dictionary = Dictionary(tokenized_texts)
    cm = CoherenceModel(
        topics=topic_wordlists,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence=coherence,
    )
    return float(cm.get_coherence())

def scan_k_nmf(
    X_tfidf,
    vectorizer,
    tokenized_texts: list[list[str]],
    k_values: list[int],
    random_state: int,
    top_n: int,
    coherence: str,
) -> pd.DataFrame:
    rows = []
    feat = vectorizer.get_feature_names_out()
    for k in k_values:
        nmf = NMF(n_components=k, random_state=random_state, init="nndsvda", max_iter=300)
        nmf.fit(X_tfidf)
        wordlists = _components_to_wordlists(nmf.components_, feat, top_n=top_n)
        coh = compute_coherence(wordlists, tokenized_texts, coherence=coherence)
        rows.append({"model": "NMF", "k": k, "coherence": coh})
    return pd.DataFrame(rows)

def scan_k_lda(
    X_tfidf,
    vectorizer,
    tokenized_texts: list[list[str]],
    k_values: list[int],
    random_state: int,
    top_n: int,
    coherence: str,
) -> pd.DataFrame:
    rows = []
    feat = vectorizer.get_feature_names_out()
    for k in k_values:
        lda = LatentDirichletAllocation(
            n_components=k,
            random_state=random_state,
            learning_method="batch",
            max_iter=20,
        )
        lda.fit(X_tfidf)
        wordlists = _components_to_wordlists(lda.components_, feat, top_n=top_n)
        coh = compute_coherence(wordlists, tokenized_texts, coherence=coherence)
        rows.append({"model": "LDA", "k": k, "coherence": coh})
    return pd.DataFrame(rows)