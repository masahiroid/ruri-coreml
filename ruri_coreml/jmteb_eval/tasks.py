"""The JMTEB tasks evaluated, and how each one is scored.

Chosen to cover every task type that matters for on-device search while
fitting overnight runs on a laptop: STS, clustering, and the two retrieval
sets whose corpora are small enough to embed six times.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ruri_coreml_convert.constants import RuriPromptPrefix

from .constants import (
    CORPUS_SPLIT,
    EVALUATION_SPLIT,
    LONG_SEQUENCE_LENGTH,
    RETRIEVAL_CUTOFF,
    SHORT_SEQUENCE_LENGTH,
)
from .data import load_split
from .metrics import kmeans_labels, ndcg_at_k, spearman_correlation, v_measure

# embed(texts, prefix, sequence_length, cache_role) -> L2-normalized rows
EmbedFunction = Callable[[list[str], RuriPromptPrefix, int, str], np.ndarray]


@dataclass(frozen=True)
class StsTask:
    name: str
    config: str
    metric_name: str = "spearman"

    def score(self, embed: EmbedFunction, row_limit: int | None) -> float:
        frame = _limited(load_split(self.config, EVALUATION_SPLIT), row_limit)
        first = embed(frame["sentence1"].tolist(), RuriPromptPrefix.NONE, SHORT_SEQUENCE_LENGTH, "sentence1")
        second = embed(frame["sentence2"].tolist(), RuriPromptPrefix.NONE, SHORT_SEQUENCE_LENGTH, "sentence2")
        similarities = (first * second).sum(axis=1)
        return spearman_correlation(similarities, frame["label"].astype(float).to_numpy())


@dataclass(frozen=True)
class ClusteringTask:
    name: str
    config: str
    sequence_length: int
    metric_name: str = "v_measure"

    def score(self, embed: EmbedFunction, row_limit: int | None) -> float:
        frame = _limited(load_split(self.config, EVALUATION_SPLIT), row_limit)
        vectors = embed(frame["text"].tolist(), RuriPromptPrefix.TOPIC, self.sequence_length, "text")
        labels = frame["label"].astype(str).to_numpy()
        predicted = kmeans_labels(vectors.astype(np.float64), len(np.unique(labels)))
        return v_measure(labels, predicted)


@dataclass(frozen=True)
class RetrievalTask:
    name: str
    query_config: str
    corpus_config: str
    metric_name: str = f"ndcg@{RETRIEVAL_CUTOFF}"

    def score(self, embed: EmbedFunction, row_limit: int | None) -> float:
        queries = _limited(load_split(self.query_config, EVALUATION_SPLIT), row_limit)
        corpus = load_split(self.corpus_config, CORPUS_SPLIT)
        if row_limit is not None:
            corpus = _corpus_subset_keeping_relevant(corpus, queries, row_limit)

        query_vectors = embed(
            queries["query"].tolist(), RuriPromptPrefix.SEARCH_QUERY, SHORT_SEQUENCE_LENGTH, "query"
        )
        document_vectors = embed(
            corpus["text"].tolist(), RuriPromptPrefix.SEARCH_DOCUMENT, LONG_SEQUENCE_LENGTH, "corpus"
        )
        document_ids = corpus["docid"].astype(str).tolist()

        scores = query_vectors @ document_vectors.T
        top = np.argsort(-scores, axis=1)[:, :RETRIEVAL_CUTOFF]
        per_query = [
            ndcg_at_k([document_ids[j] for j in top[i]], _relevant_ids(relevant), RETRIEVAL_CUTOFF)
            for i, relevant in enumerate(queries["relevant_docs"])
        ]
        return float(np.mean(per_query))


def _relevant_ids(raw) -> set[str]:
    if isinstance(raw, (list, tuple, np.ndarray)):
        return {str(x) for x in raw}
    text = str(raw)
    if text.startswith("["):
        return {str(x) for x in ast.literal_eval(text)}
    return {text}


def _limited(frame: pd.DataFrame, row_limit: int | None) -> pd.DataFrame:
    return frame if row_limit is None else frame.head(row_limit).reset_index(drop=True)


def _corpus_subset_keeping_relevant(corpus: pd.DataFrame, queries: pd.DataFrame, row_limit: int) -> pd.DataFrame:
    relevant = set().union(*(_relevant_ids(r) for r in queries["relevant_docs"]))
    is_relevant = corpus["docid"].astype(str).isin(relevant)
    return pd.concat([corpus[is_relevant], corpus[~is_relevant].head(row_limit)]).reset_index(drop=True)


TASKS = (
    StsTask("JSTS", "jsts"),
    StsTask("JSICK", "jsick"),
    ClusteringTask("livedoor_news", "livedoor_news", LONG_SEQUENCE_LENGTH),
    ClusteringTask("MewsC-16-ja", "mewsc16_ja", SHORT_SEQUENCE_LENGTH),
    RetrievalTask("NLP Journal title→abs", "nlp_journal_title_abs-query", "nlp_journal_title_abs-corpus"),
    RetrievalTask("JaGovFaqs-22k", "jagovfaqs_22k-query", "jagovfaqs_22k-corpus"),
)
