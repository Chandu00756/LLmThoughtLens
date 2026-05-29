"""Numerical helpers used across the package.

Everything here works on plain NumPy arrays so that core modules
(SAE inference, graph maths, probes) do not import torch unconditionally.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along *axis*."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def topk_indices(x: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Return the indices of the top-*k* values along *axis* (descending)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, x.shape[axis])
    # argpartition is faster than full argsort for large arrays
    idx = np.argpartition(-x, kth=k - 1, axis=axis)
    idx = np.take(idx, np.arange(k), axis=axis)
    # sort the partial slice descending
    sorted_part = np.take_along_axis(
        np.argsort(-np.take_along_axis(x, idx, axis=axis), axis=axis),
        np.arange(k).reshape([-1 if d == axis else 1 for d in range(x.ndim)]),
        axis=axis,
    )
    return np.take_along_axis(idx, sorted_part, axis=axis)


def topk_mask(x: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Zero out everything except the top-*k* magnitudes along *axis*.

    Returns a new array of the same shape and dtype.  Negative values that
    rank in the top-k by raw value (not magnitude) are retained — the TopK
    SAE keeps positive activations only because the encoder uses ReLU first.
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    k = min(k, x.shape[axis])
    if k == x.shape[axis]:
        return x.copy()
    idx = np.argpartition(-x, kth=k - 1, axis=axis)
    keep_idx = np.take(idx, np.arange(k), axis=axis)
    out = np.zeros_like(x)
    np.put_along_axis(out, keep_idx, np.take_along_axis(x, keep_idx, axis=axis), axis=axis)
    return out


def l2_normalise(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Return *x* divided by its L2 norm along *axis*."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def pca_2d(x: np.ndarray) -> np.ndarray:
    """Return the 2-component PCA projection of *x* (rows are samples).

    Pure-NumPy SVD-based PCA so the residual-stream view does not depend
    on scikit-learn.
    """
    x = np.asarray(x, dtype=np.float64)
    centred = x - x.mean(axis=0, keepdims=True)
    # economy SVD; first 2 right singular vectors are the top-2 PCs.
    _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
    components = vh[:2]
    return (centred @ components.T).astype(np.float32)


def cumulative_probability_mass(probs: np.ndarray) -> np.ndarray:
    """Return cumulative sum of a probability vector sorted descending."""
    return np.cumsum(np.sort(probs)[::-1])
