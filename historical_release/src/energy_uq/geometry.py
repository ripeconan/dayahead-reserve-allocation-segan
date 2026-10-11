"""Common multivariate OAS conformal geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ellipsoid:
    center: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    threshold: float


def conformal_quantile(values: np.ndarray, coverage: float) -> float:
    """Return the finite-sample upper conformal quantile."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite vector")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie strictly between zero and one")
    rank = min(len(array), math.ceil((len(array) + 1) * coverage))
    return float(np.partition(array, rank - 1)[rank - 1])


def oas_covariance(values: np.ndarray) -> np.ndarray:
    """Compute oracle-approximating shrinkage covariance."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError("values must be a finite two-dimensional sample")
    centered = array - array.mean(axis=0)
    n_rows, dimension = centered.shape
    empirical = centered.T @ centered / n_rows
    mu = np.trace(empirical) / dimension
    alpha = np.mean(empirical**2)
    denominator = (n_rows + 1.0) * (alpha - mu**2 / dimension)
    shrinkage = 1.0 if denominator <= 0 else min((alpha + mu**2) / denominator, 1.0)
    covariance = (1.0 - shrinkage) * empirical
    covariance.flat[:: dimension + 1] += shrinkage * mu
    covariance.flat[:: dimension + 1] += max(mu, 1.0) * 1e-10
    return covariance


def squared_distance(
    values: np.ndarray, center: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    delta = np.asarray(values, dtype=float) - np.asarray(center, dtype=float)
    return np.einsum("ni,ij,nj->n", delta, precision, delta)


def fit_ellipsoid(values: np.ndarray, coverage: float) -> Ellipsoid:
    array = np.asarray(values, dtype=float)
    center = array.mean(axis=0)
    covariance = oas_covariance(array)
    precision = np.linalg.pinv(covariance, hermitian=True)
    scores = squared_distance(array, center, precision)
    threshold = conformal_quantile(scores, coverage)
    return Ellipsoid(center, covariance, precision, threshold)


def log_volume_proxy(model: Ellipsoid) -> float:
    sign, log_determinant = np.linalg.slogdet(model.covariance)
    if sign <= 0:
        raise ValueError("covariance must have a positive determinant")
    dimension = len(model.center)
    return 0.5 * log_determinant + 0.5 * dimension * math.log(model.threshold)

