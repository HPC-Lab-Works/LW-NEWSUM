"""Extractive reference systems: Lead-3 (See et al., 2017) and TextRank (Mihalcea and Tarau, 2004)."""
from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import split_sentences


def lead_k(text: str, k: int = 3) -> str:
    return " ".join(split_sentences(text)[:k])


def textrank(text: str, k: int = 3, damping: float = 0.85) -> str:
    """Sentence-level TextRank over a TF-IDF cosine-similarity graph.

    The top-k sentences are returned in their original document order.
    """
    sents = split_sentences(text)
    if len(sents) <= k:
        return " ".join(sents)
    try:
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(sents)
    except ValueError:  # only stop words
        return " ".join(sents[:k])
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0.0)
    graph = nx.from_numpy_array(sim)
    try:
        scores = nx.pagerank(graph, alpha=damping, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        return " ".join(sents[:k])
    top = sorted(sorted(scores, key=scores.get, reverse=True)[:k])
    return " ".join(sents[i] for i in top)


def k_for_dataset(dataset: str) -> int:
    # XSum references are single sentences; Lead-1 is the conventional extractive reference there.
    return 1 if dataset == "xsum" else 3
