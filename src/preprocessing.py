from __future__ import annotations
import re
from typing import Iterable, List
import spacy

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"\b[xX]{2,}\b")
MULTI_DOTS_RE = re.compile(r"\.{2,}")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

def basic_clean(text: str) -> str:
    text = text.strip()
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PLACEHOLDER_RE.sub(" ", text)
    text = MULTI_DOTS_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text

def spacy_lemmatize(
    texts: Iterable[str],
    min_token_len: int = 3,
    batch_size: int = 500,
) -> List[str]:

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    cleaned_docs: List[str] = []

    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = []
        for t in doc:
            if t.is_stop:
                continue
            if not t.is_alpha:
                continue
            lemma = t.lemma_.lower().strip()
            if len(lemma) < min_token_len:
                continue
            tokens.append(lemma)
        cleaned_docs.append(" ".join(tokens))

    return cleaned_docs