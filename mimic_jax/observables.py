"""Model-neutral catalogue summaries shared by SAGE16 and SHARK.

These ordinary NumPy reductions sit outside differentiable physics kernels.
They make selections, bins, units, and zero handling identical across model
adapters so a comparison cannot accidentally change the observable definition.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinnedRelation:
    """Median and central percentile interval in bins of a predictor."""

    bin_edges: np.ndarray
    bin_centres: np.ndarray
    counts: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class BinnedFraction:
    """Selected fraction and integer counts in bins of a predictor."""

    bin_edges: np.ndarray
    bin_centres: np.ndarray
    counts: np.ndarray
    selected_counts: np.ndarray
    fraction: np.ndarray


def _validated_vectors(predictor, response=None):
    x = np.asarray(predictor, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("predictor must be one-dimensional")
    if response is None:
        return x
    y = np.asarray(response, dtype=np.float64)
    if y.ndim != 1 or y.shape != x.shape:
        raise ValueError("response must be one-dimensional and match predictor")
    return x, y


def _validated_edges(bin_edges):
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges must be a strictly increasing one-dimensional array")
    return edges


def binned_relation(
    predictor,
    response,
    *,
    bin_edges,
    percentiles=(16.0, 84.0),
) -> BinnedRelation:
    """Return a finite-value median relation with explicit empty bins."""

    x, y = _validated_vectors(predictor, response)
    edges = _validated_edges(bin_edges)
    lower_percentile, upper_percentile = percentiles
    if not 0.0 <= lower_percentile <= 50.0 <= upper_percentile <= 100.0:
        raise ValueError("percentiles must bracket the median within [0, 100]")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    indices = np.searchsorted(edges, x, side="right") - 1
    number_of_bins = edges.size - 1
    counts = np.zeros(number_of_bins, dtype=np.int64)
    median = np.full(number_of_bins, np.nan)
    lower = np.full(number_of_bins, np.nan)
    upper = np.full(number_of_bins, np.nan)
    for index in range(number_of_bins):
        selected = y[indices == index]
        counts[index] = selected.size
        if selected.size:
            lower[index], median[index], upper[index] = np.percentile(
                selected, (lower_percentile, 50.0, upper_percentile)
            )
    return BinnedRelation(
        bin_edges=edges,
        bin_centres=edges[:-1] + 0.5 * np.diff(edges),
        counts=counts,
        median=median,
        lower=lower,
        upper=upper,
    )


def binned_selected_fraction(predictor, selected, *, bin_edges) -> BinnedFraction:
    """Return a boolean-selected fraction with explicit empty bins."""

    x = _validated_vectors(predictor)
    mask = np.asarray(selected, dtype=bool)
    if mask.ndim != 1 or mask.shape != x.shape:
        raise ValueError("selected must be one-dimensional and match predictor")
    edges = _validated_edges(bin_edges)
    finite = np.isfinite(x)
    counts, _ = np.histogram(x[finite], bins=edges)
    selected_counts, _ = np.histogram(x[finite & mask], bins=edges)
    fraction = np.full(counts.shape, np.nan, dtype=np.float64)
    populated = counts > 0
    fraction[populated] = selected_counts[populated] / counts[populated]
    return BinnedFraction(
        bin_edges=edges,
        bin_centres=edges[:-1] + 0.5 * np.diff(edges),
        counts=counts.astype(np.int64),
        selected_counts=selected_counts.astype(np.int64),
        fraction=fraction,
    )
