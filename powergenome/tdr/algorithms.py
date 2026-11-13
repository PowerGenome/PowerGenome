"""K-means and k-medoids clustering of candidate periods.

Both backends return one assignment per input column, the number of periods in
each cluster, and one distinct representative period per cluster. K-means uses
scikit-learn; k-medoids is implemented here with alternating (Voronoi) medoids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

CLUSTERING_METHODS = ("kmeans", "kmedoids")

_MAX_MEDOID_ITERATIONS = 100


@dataclass(frozen=True)
class ClusteringResult:
    """Cluster assignment of the candidate periods.

    Attributes
    ----------
    assignments : np.ndarray
        0-based cluster index for each of the ``n_periods`` input columns.
    counts : np.ndarray
        Number of periods assigned to each cluster.
    representatives : np.ndarray
        0-based index of each cluster's representative period, one per cluster and
        always distinct.
    """

    assignments: np.ndarray
    counts: np.ndarray
    representatives: np.ndarray


def cluster_periods(
    matrix: np.ndarray,
    n_clusters: int,
    method: str = "kmeans",
    n_restarts: int = 10,
    random_state: int = 42,
) -> ClusteringResult:
    """Cluster the columns of a candidate-period matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Shape ``(n_features, n_periods)``. Columns are the observations.
    n_clusters : int
        Number of clusters to form.
    method : str, optional
        ``"kmeans"`` (scikit-learn, k-means++) or ``"kmedoids"`` (alternating
        medoids seeded with k-means++ over the distance matrix), by default
        "kmeans".
    n_restarts : int, optional
        Number of independent initializations, by default 10.
    random_state : int, optional
        Seed for initialization, by default 42.

    Returns
    -------
    ClusteringResult
        Assignments, cluster counts, and distinct representative periods.
    """
    if method not in CLUSTERING_METHODS:
        raise ValueError(
            f"Unknown clustering method '{method}'. Expected one of {CLUSTERING_METHODS}."
        )
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be at least 1, got {n_clusters}.")
    if n_clusters > matrix.shape[1]:
        raise ValueError(
            f"Cannot form {n_clusters} clusters from {matrix.shape[1]} candidate "
            "periods."
        )
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be at least 1, got {n_restarts}.")

    points = matrix.T
    if method == "kmeans":
        return _kmeans(points, n_clusters, n_restarts, random_state)
    return _kmedoids(points, n_clusters, n_restarts, random_state)


def _kmeans(
    points: np.ndarray, n_clusters: int, n_restarts: int, random_state: int
) -> ClusteringResult:
    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_restarts,
        init="k-means++",
        random_state=random_state,
    )
    labels = model.fit_predict(points)
    centers = np.array(model.cluster_centers_, copy=True)
    labels = _repair_empty_clusters(points, labels, centers, n_clusters)
    centers = _centers(points, labels, n_clusters)
    representatives = _nearest_distinct(points, centers, n_clusters)
    return ClusteringResult(
        assignments=labels,
        counts=np.bincount(labels, minlength=n_clusters),
        representatives=representatives,
    )


def _centers(points: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    return np.vstack(
        [points[labels == cluster].mean(axis=0) for cluster in range(n_clusters)]
    )


def _repair_empty_clusters(
    points: np.ndarray, labels: np.ndarray, centers: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Move the member farthest from its center into every empty cluster."""
    labels = labels.copy()
    for empty in np.flatnonzero(np.bincount(labels, minlength=n_clusters) == 0):
        counts = np.bincount(labels, minlength=n_clusters)
        donor = int(np.argmax(counts))
        if counts[donor] <= 1:
            raise ValueError(
                "Clustering produced more empty clusters than it can repair."
            )
        members = np.flatnonzero(labels == donor)
        distances = np.linalg.norm(points[members] - centers[donor], axis=1)
        labels[members[np.argmax(distances)]] = empty
        centers[donor] = points[labels == donor].mean(axis=0)
        centers[empty] = points[labels == empty].mean(axis=0)
    return labels


def _nearest_distinct(
    points: np.ndarray, centers: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Pick each cluster's nearest period, skipping periods already chosen."""
    chosen: set[int] = set()
    representatives = []
    for cluster in range(n_clusters):
        distances = np.linalg.norm(points - centers[cluster], axis=1)
        for candidate in np.argsort(distances, kind="stable"):
            if int(candidate) not in chosen:
                representatives.append(int(candidate))
                chosen.add(int(candidate))
                break
        else:
            raise ValueError(
                "Clustering did not identify enough distinct candidate periods."
            )
    return np.array(representatives)


def _kmedoids(
    points: np.ndarray, n_clusters: int, n_restarts: int, random_state: int
) -> ClusteringResult:
    """Alternating medoids: assign to the nearest medoid, then re-pick each medoid
    as the member minimizing its cluster's within-cluster distance. The best of
    ``n_restarts`` k-means++ seeded runs is kept."""
    distances = _distance_matrix(points)
    rng = np.random.default_rng(random_state)

    best = None
    for _ in range(n_restarts):
        medoids = _kmeans_plus_plus(distances, n_clusters, rng)
        medoids, assignments = _alternating_medoids(distances, n_clusters, medoids)
        cost = distances[np.arange(len(assignments)), medoids[assignments]].sum()
        if best is None or cost < best[0] - 1e-9:
            best = (cost, medoids, assignments)

    _, medoids, assignments = best
    return ClusteringResult(
        assignments=assignments,
        counts=np.bincount(assignments, minlength=n_clusters),
        representatives=medoids,
    )


def _alternating_medoids(
    distances: np.ndarray, n_clusters: int, medoids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    medoids = np.array(medoids, dtype=int)
    assignments = _assign_to_medoids(distances, medoids)
    for _ in range(_MAX_MEDOID_ITERATIONS):
        updated = medoids.copy()
        for cluster in range(n_clusters):
            members = np.flatnonzero(assignments == cluster)
            within = distances[np.ix_(members, members)].sum(axis=1)
            updated[cluster] = members[np.argmin(within)]
        if np.array_equal(updated, medoids):
            break
        medoids = updated
        assignments = _assign_to_medoids(distances, medoids)
    return medoids, assignments


def _assign_to_medoids(distances: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    """Assign every period to its nearest medoid.

    Medoids are pinned to their own cluster so that identical profiles cannot
    leave a cluster empty.
    """
    assignments = np.argmin(distances[:, medoids], axis=1)
    assignments[medoids] = np.arange(len(medoids))
    return assignments


def _distance_matrix(points: np.ndarray) -> np.ndarray:
    """Euclidean distance between every pair of candidate periods."""
    squared_norms = np.einsum("ij,ij->i", points, points)
    squared = squared_norms[:, None] + squared_norms[None, :] - 2 * points @ points.T
    return np.sqrt(np.maximum(squared, 0))


def _kmeans_plus_plus(
    distances: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    """k-means++ seeding over a precomputed distance matrix."""
    n_points = distances.shape[0]
    selected = [int(rng.integers(n_points))]
    closest = distances[:, selected[0]] ** 2

    for _ in range(1, n_clusters):
        total = closest.sum()
        if total == 0:
            candidate = next(i for i in range(n_points) if i not in selected)
        else:
            candidate = int(rng.choice(n_points, p=closest / total))
        selected.append(candidate)
        closest = np.minimum(closest, distances[:, candidate] ** 2)

    return np.array(selected)
