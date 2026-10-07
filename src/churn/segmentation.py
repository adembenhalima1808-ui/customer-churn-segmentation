"""K-means customer segmentation: scaling, choosing k, and profiling segments."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def _kmeans(k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)


def _fit_quietly(model: KMeans, scaled):
    # On some macOS numpy builds the BLAS matmul raises spurious floating-point
    # flags on finite, bounded input. Results are checked for finiteness in the
    # tests, so the flags are silenced here only.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return model.fit_predict(scaled)


@dataclass
class KSweep:
    k_values: list
    inertia: list
    silhouette: list

    def best_k(self) -> int:
        """Return the k with the highest silhouette score."""
        best_index = max(range(len(self.silhouette)), key=lambda i: self.silhouette[i])
        return self.k_values[best_index]


def scale(matrix: pd.DataFrame) -> tuple:
    """Standardise features so that no single unit (dollars, months) dominates distance."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return scaled, scaler


def sweep_k(scaled, k_values=range(2, 9)) -> KSweep:
    """Fit k-means for each k and record inertia and silhouette."""
    inertia, silhouettes = [], []
    for k in k_values:
        model = _kmeans(k)
        labels = _fit_quietly(model, scaled)
        inertia.append(float(model.inertia_))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            silhouettes.append(float(silhouette_score(scaled, labels)))
    return KSweep(k_values=list(k_values), inertia=inertia, silhouette=silhouettes)


def fit_segments(scaled, k: int) -> tuple:
    """Fit the final model and return (model, labels)."""
    model = _kmeans(k)
    labels = _fit_quietly(model, scaled)
    return model, labels


def profile_segments(matrix: pd.DataFrame, labels) -> pd.DataFrame:
    """Mean of each feature per segment, plus segment size, in original units."""
    frame = matrix.copy()
    frame["segment"] = labels
    summary = frame.groupby("segment").agg(["mean"])
    summary.columns = [col for col, _ in summary.columns]
    summary["size"] = frame.groupby("segment").size()
    summary["share"] = summary["size"] / summary["size"].sum()
    return summary.round(2)
