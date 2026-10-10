"""Benchmark metrics implemented with numpy only (scipy/scikit-learn wheels do
not load on the macOS beta this was developed on).
"""

from __future__ import annotations

import math

import numpy as np

from .constants import KMEANS_MAX_ITERATIONS, KMEANS_RESTARTS, KMEANS_TOLERANCE, RANDOM_SEED


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_average_ranks(a), _average_ranks(b))[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0
        start = end + 1
    return ranks


def kmeans_labels(vectors: np.ndarray, cluster_count: int) -> np.ndarray:
    """Best-inertia result over several k-means++ restarts."""
    rng = np.random.default_rng(RANDOM_SEED)
    best_labels, best_inertia = None, math.inf
    for _ in range(KMEANS_RESTARTS):
        labels, inertia = _kmeans_once(vectors, cluster_count, rng)
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels


def _kmeans_once(vectors: np.ndarray, cluster_count: int, rng: np.random.Generator):
    centers = _kmeans_plus_plus(vectors, cluster_count, rng)
    previous_inertia = math.inf
    for _ in range(KMEANS_MAX_ITERATIONS):
        distances = _squared_distances(vectors, centers)
        labels = distances.argmin(axis=1)
        inertia = float(distances[np.arange(len(vectors)), labels].sum())
        for k in range(cluster_count):
            members = vectors[labels == k]
            if len(members):
                centers[k] = members.mean(axis=0)
        if previous_inertia - inertia <= KMEANS_TOLERANCE * max(previous_inertia, 1.0):
            break
        previous_inertia = inertia
    return labels, inertia


def _kmeans_plus_plus(vectors: np.ndarray, cluster_count: int, rng: np.random.Generator) -> np.ndarray:
    centers = [vectors[rng.integers(len(vectors))]]
    for _ in range(1, cluster_count):
        nearest = _squared_distances(vectors, np.array(centers)).min(axis=1)
        centers.append(vectors[rng.choice(len(vectors), p=nearest / nearest.sum())])
    return np.array(centers, dtype=np.float64)


def _squared_distances(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.maximum(
        (vectors**2).sum(axis=1)[:, None] - 2 * vectors @ centers.T + (centers**2).sum(axis=1)[None, :],
        0.0,
    )


def v_measure(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    _, true_index = np.unique(true_labels, return_inverse=True)
    _, predicted_index = np.unique(predicted_labels, return_inverse=True)
    contingency = np.zeros((true_index.max() + 1, predicted_index.max() + 1))
    np.add.at(contingency, (true_index, predicted_index), 1)
    total = contingency.sum()
    joint = contingency / total
    true_marginal = joint.sum(axis=1)
    predicted_marginal = joint.sum(axis=0)
    true_entropy = _entropy(true_marginal)
    predicted_entropy = _entropy(predicted_marginal)
    nonzero = joint > 0
    mutual_information = float(
        (joint[nonzero] * np.log(joint[nonzero] / np.outer(true_marginal, predicted_marginal)[nonzero])).sum()
    )
    homogeneity = 1.0 if true_entropy == 0 else mutual_information / true_entropy
    completeness = 1.0 if predicted_entropy == 0 else mutual_information / predicted_entropy
    if homogeneity + completeness == 0:
        return 0.0
    return 2 * homogeneity * completeness / (homogeneity + completeness)


def _entropy(probabilities: np.ndarray) -> float:
    nonzero = probabilities[probabilities > 0]
    return float(-(nonzero * np.log(nonzero)).sum())


def ndcg_at_k(ranked_doc_ids: list[str], relevant_doc_ids: set[str], k: int) -> float:
    gains = [1.0 if doc_id in relevant_doc_ids else 0.0 for doc_id in ranked_doc_ids[:k]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant_doc_ids), k)))
    return dcg / ideal if ideal else 0.0
