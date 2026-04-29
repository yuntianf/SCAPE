from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math
from typing import Tuple

# -------------------------
# Utils
# -------------------------

def set_seed(seed: int = 0):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def zscore_fit_transform(x: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: x_scaled, mu, sd
    """
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + eps
    return (x - mu) / sd, mu, sd


def zscore_transform(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd

def pairwise_sqeuclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    AA = np.sum(A * A, axis=1, keepdims=True)
    BB = np.sum(B * B, axis=1, keepdims=True).T
    return AA + BB - 2.0 * (A @ B.T)

def cost_scale(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    pos = C[C > 0]
    med = np.median(pos) if pos.size else 1.0
    return C / (med + eps)


# =========================
# Gradient Reversal (GRL)
# =========================

@tf.custom_gradient
def _grl_op(x,lambd):
    def grad(dy):
        return -lambd * dy, None
    return x, grad

class GradientReversal(layers.Layer):
    def __init__(self, lambd=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambd = float(lambd)

    def call(self, x):
        return _grl_op(x, tf.constant(self.lambd, x.dtype))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"lambd": self.lambd})
        return cfg
    
# -----------------------------
# Preprocessing helpers
# -----------------------------
def col_scale(X,eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + eps
    return (X - mu) / sd    

def log1p_standardize(X, eps=1e-8):
    X = np.log1p(X)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + eps
    return (X - mu) / sd, (mu, sd)

def clr_transform_proportions(P, pseudocount=1e-6, eps=1e-12):
    P = np.clip(P, 0.0, 1.0)
    P = P + pseudocount
    P = P / (P.sum(axis=1, keepdims=True) + eps)
    logP = np.log(P + eps)
    gm = logP.mean(axis=1, keepdims=True)
    X = logP - gm
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + eps
    return (X - mu) / sd, (mu, sd)

def apply_standardization(X, stats, eps=1e-8):
    mu, sd = stats
    return (X - mu) / (sd + eps)

def row_mass_sparsify(P, keep_mass=0.8, eps=1e-12):
    """
    Sparsify a coupling matrix P by keeping the largest entries in each row
    whose cumulative mass reaches 'keep_mass' (e.g. 0.8), then renormalize.

    Parameters
    ----------
    P : array (n_src, n_tgt)
        OT coupling matrix (non-negative).
    keep_mass : float in (0,1]
        Fraction of row mass to retain.
    eps : float
        Numerical stability constant.

    Returns
    -------
    P_sparse : array (n_src, n_tgt)
        Row-sparsified and renormalized coupling.
    """

    P = np.asarray(P, dtype=np.float64)
    n_src, n_tgt = P.shape

    P_sparse = np.zeros_like(P)

    for i in range(n_src):
        row = P[i]

        if row.sum() <= eps:
            continue  # leave as zero row (rare edge case)

        # Sort indices by descending weight
        idx_sorted = np.argsort(-row)

        # Cumulative mass
        cum_mass = np.cumsum(row[idx_sorted])
        total_mass = cum_mass[-1]

        # Find cutoff index
        cutoff = np.searchsorted(cum_mass, keep_mass * total_mass)

        # Indices to keep
        keep_idx = idx_sorted[:cutoff + 1]

        # Keep and renormalize
        kept_values = row[keep_idx]
        kept_values = kept_values / (kept_values.sum() + eps)

        P_sparse[i, keep_idx] = kept_values

    return P_sparse
