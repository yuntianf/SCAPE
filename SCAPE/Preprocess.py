# ====== Imports ======
import numpy as np

# =========================
# Preprocessing utilities
# =========================

class GeneZScaler:
    """Per-gene z-scaler: fit on TRAIN only; apply to any split."""
    def __init__(self, eps=1e-3):
        self.mu = None
        self.sd = None
        self.eps = eps

    def fit(self, Y_train):
        self.mu = Y_train.mean(axis=0, keepdims=True).astype(np.float32)
        self.sd = Y_train.std(axis=0, keepdims=True).astype(np.float32)
        self.sd = np.maximum(self.sd, self.eps)
        return self

    def transform(self, Y):
        return (Y - self.mu) / self.sd

    def inverse(self, Yz):
        return Yz * self.sd + self.mu


class NeighborXTransformer:
    """
    Compose neighbor counts X (n,m') into:
      - per-cell proportions P = X / sum(X_i)
      - size covariate s = log1p(sum(X_i))
    Then standardize columns on TRAIN split.
    """
    def __init__(self, eps=1e-3):
        self.mu = None
        self.sd = None
        self.eps = eps
        self.out_dim = None

    def _compose(self, X):
        N = X.sum(axis=1, keepdims=True)
        P = X / np.maximum(N, 1.0)
        s = np.log1p(N)
        PX = np.hstack([P, s])
        return PX

    def fit(self, X_train):
        PX = self._compose(X_train)
        self.mu = PX.mean(axis=0, keepdims=True).astype(np.float32)
        self.sd = PX.std(axis=0, keepdims=True).astype(np.float32)
        self.sd = np.maximum(self.sd, self.eps)
        self.out_dim = PX.shape[1]
        return self

    def transform(self, X):
        PX = self._compose(X)
        return (PX - self.mu) / self.sd


class SlideOneHotEncoder:
    """One-hot for slide IDs; fit on TRAIN levels only."""
    def fit(self, slide_ids):
        self.levels_ = np.unique(slide_ids)
        self.index_ = {lv: i for i, lv in enumerate(self.levels_)}
        self.n_slides_ = len(self.levels_)
        return self

    def transform(self, slide_ids):
        # Assert no unseen slides in val (design folds within slides)
        if np.setdiff1d(np.unique(slide_ids), self.levels_).size > 0:
            unseen = np.setdiff1d(np.unique(slide_ids), self.levels_)
            raise ValueError(f"Unseen slide IDs in validation: {unseen}. "
                             f"Build folds within slides so each fold has all slides.")
        idx = np.array([self.index_[lv] for lv in slide_ids], dtype=int)
        onehot = np.zeros((len(idx), self.n_slides_), dtype=np.float32)
        onehot[np.arange(len(idx)), idx] = 1.0
        return onehot
