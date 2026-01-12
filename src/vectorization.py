from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def build_tfidf(
    texts: list[str],
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.9,
    max_features: int | None = 50000,
):
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def build_sbert_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings