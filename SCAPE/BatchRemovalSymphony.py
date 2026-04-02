import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from sklearn.decomposition import PCA
import harmonypy as hm


def _as_2d_float(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D: cells x features")
    return X


def _l2_normalize_columns(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Input: d x n
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    norms = np.maximum(norms, eps)
    return M / norms


def _one_hot_from_batches(batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X: c x m one-hot batch matrix
        categories: unique batch labels in order
    """
    batch = np.asarray(batch)
    categories, inv = np.unique(batch, return_inverse=True)
    X = np.zeros((len(categories), len(batch)), dtype=float)
    X[inv, np.arange(len(batch))] = 1.0
    return X, categories


def _soft_cluster(Y_cos: np.ndarray, Z_cos: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Symphony soft assignment.

    Y_cos: d x k   reference centroids, L2-normalized by column
    Z_cos: d x m   query cells, L2-normalized by column

    Returns:
        R: k x m, columns sum to 1
    """
    # cosine similarity between centroids and cells
    sim = Y_cos.T @ Z_cos  # k x m

    # Eq. (9) in Symphony paper:
    # R(k, i) ∝ exp( -2/sigma * (1 - sim) )
    logits = (-2.0 / sigma) * (1.0 - sim)

    # stable softmax over clusters for each cell
    logits = logits - logits.max(axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    R = exp_logits / exp_logits.sum(axis=0, keepdims=True)
    return R


@dataclass
class SymphonyReference:
    # scaling from untreated reference only
    mean_: np.ndarray               # p
    std_: np.ndarray                # p

    # PCA learned on untreated reference only
    pca_components_: np.ndarray     # d x p  (sklearn components_)
    explained_variance_: np.ndarray # d

    # embeddings
    Z_ref_orig: np.ndarray          # d x n_ref  (reference PCA before Harmony)
    Z_ref_harmony: np.ndarray       # d x n_ref  (reference harmonized embedding)

    # Harmony/Symphony quantities
    R_ref: np.ndarray               # k x n_ref
    Y_cos: np.ndarray               # d x k   L2-normalized reference centroids
    Nr: np.ndarray                  # k
    C: np.ndarray                   # k x d

    # bookkeeping
    ref_batches_: np.ndarray        # unique reference batch labels
    ref_index_: np.ndarray          # indices of untreated cells in original order

    # settings
    n_pcs: int
    sigma: float
    ridge_lambda: float


class Symphony:
    """
    Faithful Symphony-style reference build + query map for generic cell-feature matrices.

    Assumptions:
    - X is cells x features.
    - 'treated == False' cells define the reference.
    - Harmony is fit only on untreated/reference cells.
    - Query cells are mapped without moving the reference.

    Notes:
    - This is Symphony's core mapping logic:
      projection -> soft assignment -> MoE correction with Nr and C.
    - It does NOT re-implement Harmony training itself; it uses harmonypy for that step.
    """

    def __init__(
        self,
        n_pcs: int = 20,
        n_clusters: Optional[int] = None,
        sigma: float = 0.1,
        harmony_theta: float = 2.0,
        harmony_lambda: float = 1.0,
        harmony_max_iter: int = 20,
        random_state: int = 0,
        scale_clip: Optional[float] = None,
    ):
        self.n_pcs = int(n_pcs)
        self.n_clusters = n_clusters
        self.sigma = float(sigma)
        self.harmony_theta = float(harmony_theta)
        self.harmony_lambda = float(harmony_lambda)
        self.harmony_max_iter = int(harmony_max_iter)
        self.random_state = int(random_state)
        self.scale_clip = scale_clip
        self.ref_: Optional[SymphonyReference] = None

    def _fit_reference_scaler(self, X_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_ = X_ref.mean(axis=0)
        std_ = X_ref.std(axis=0, ddof=0)
        std_[std_ == 0] = 1.0
        return mean_, std_

    def _transform_with_ref_scaler(self, X: np.ndarray, mean_: np.ndarray, std_: np.ndarray) -> np.ndarray:
        Xs = (X - mean_) / std_
        if self.scale_clip is not None:
            Xs = np.clip(Xs, -self.scale_clip, self.scale_clip)
        return Xs

    def _run_reference_harmony(
        self,
        Z_ref: np.ndarray,    # n_ref x d
        batch_ref: np.ndarray
    ) -> Dict[str, Any]:
        meta = pd.DataFrame({"batch": batch_ref.astype(str)})

        k = self.n_clusters
        if k is None:
            k = min(100, max(2, Z_ref.shape[0] // 30))

        ho = hm.run_harmony(
            data_mat=Z_ref,
            meta_data=meta,
            vars_use=["batch"],
            theta=self.harmony_theta,
            lamb=self.harmony_lambda,
            nclust=k,
            max_iter_harmony=self.harmony_max_iter,
            random_state=self.random_state,
        )

        Z_corr = np.asarray(ho.Z_corr, dtype=float)
        Z_orig = np.asarray(ho.Z_orig, dtype=float)
        R_ref = np.asarray(ho.R, dtype=float)

        n_ref, d = Z_ref.shape

        # Force everything into Symphony's expected orientation:
        # Z_* : d x n_ref
        # R   : k x n_ref

        # Z_corr
        if Z_corr.shape == (d, n_ref):
            pass
        elif Z_corr.shape == (n_ref, d):
            Z_corr = Z_corr.T
        else:
            raise ValueError(
                f"Unexpected ho.Z_corr shape {Z_corr.shape}; expected {(d, n_ref)} or {(n_ref, d)}"
            )

        # Z_orig
        if Z_orig.shape == (d, n_ref):
            pass
        elif Z_orig.shape == (n_ref, d):
            Z_orig = Z_orig.T
        else:
            raise ValueError(
                f"Unexpected ho.Z_orig shape {Z_orig.shape}; expected {(d, n_ref)} or {(n_ref, d)}"
            )

        # R_ref
        if R_ref.shape == (k, n_ref):
            pass
        elif R_ref.shape == (n_ref, k):
            R_ref = R_ref.T
        else:
            raise ValueError(
                f"Unexpected ho.R shape {R_ref.shape}; expected {(k, n_ref)} or {(n_ref, k)}"
            )

        return {
            "Z_corr": Z_corr,   # d x n_ref
            "Z_orig": Z_orig,   # d x n_ref
            "R_ref": R_ref,     # k x n_ref
            "k": R_ref.shape[0],
        }

    def fit(self, X: np.ndarray, batch: np.ndarray, treated: np.ndarray) -> "Symphony":
        """
        Build untreated reference and compress it into Symphony minimal elements.

        Parameters
        ----------
        X : array, shape (n_cells, n_features)
            Raw or at least not globally leak-scaled matrix preferred.
        batch : array, shape (n_cells,)
            Batch label for each cell.
        treated : bool array, shape (n_cells,)
            False = untreated/reference, True = treated/query.
        """
        X = _as_2d_float(X)
        batch = np.asarray(batch)
        treated = np.asarray(treated).astype(bool)

        if len(batch) != X.shape[0] or len(treated) != X.shape[0]:
            raise ValueError("batch and treated must have length n_cells")

        ref_mask = ~treated
        if ref_mask.sum() < 2:
            raise ValueError("Need at least 2 untreated/reference cells")

        X_ref = X[ref_mask]
        batch_ref = batch[ref_mask]
        ref_index = np.where(ref_mask)[0]

        # 1) reference-only scaling
        mean_, std_ = self._fit_reference_scaler(X_ref)
        X_ref_scaled = self._transform_with_ref_scaler(X_ref, mean_, std_)

        # 2) PCA on untreated reference only
        n_pcs = min(self.n_pcs, X_ref_scaled.shape[0], X_ref_scaled.shape[1])
        pca = PCA(n_components=n_pcs, svd_solver="full", random_state=self.random_state)
        Z_ref_cells_by_pc = pca.fit_transform(X_ref_scaled)  # n_ref x d

        # 3) Harmony on untreated reference only
        harm = self._run_reference_harmony(Z_ref_cells_by_pc, batch_ref)

        Z_ref_orig = harm["Z_orig"]       # d x n_ref
        Z_ref_harmony = harm["Z_corr"]    # d x n_ref
        R_ref = harm["R_ref"]             # k x n_ref

        # 4) Symphony compression
        # centroids from harmonized reference, then L2-normalize columns
        centroid_sums = Z_ref_harmony @ R_ref.T   # d x k
        Y_cos = _l2_normalize_columns(centroid_sums)

        # reference compression terms
        Nr = R_ref.sum(axis=1)            # k
        C = R_ref @ Z_ref_harmony.T       # k x d

        self.ref_ = SymphonyReference(
            mean_=mean_,
            std_=std_,
            pca_components_=pca.components_,  # d x p
            explained_variance_=pca.explained_variance_,
            Z_ref_orig=Z_ref_orig,
            Z_ref_harmony=Z_ref_harmony,
            R_ref=R_ref,
            Y_cos=Y_cos,
            Nr=Nr,
            C=C,
            ref_batches_=np.unique(batch_ref),
            ref_index_=ref_index,
            n_pcs=n_pcs,
            sigma=self.sigma,
            ridge_lambda=self.harmony_lambda,
        )
        return self

    def _project_query(self, X_query: np.ndarray) -> np.ndarray:
        ref = self.ref_
        if ref is None:
            raise RuntimeError("Call fit() first")

        Xq_scaled = self._transform_with_ref_scaler(X_query, ref.mean_, ref.std_)
        # sklearn components_ is d x p, so projected query as cells x d is X @ components_.T
        Zq_cells_by_pc = Xq_scaled @ ref.pca_components_.T
        return Zq_cells_by_pc.T  # d x m, to match Symphony notation

    def _map_query(self, X_query: np.ndarray, batch_query: np.ndarray) -> Dict[str, np.ndarray]:
        ref = self.ref_
        if ref is None:
            raise RuntimeError("Call fit() first")

        m = X_query.shape[0]
        if m == 0:
            return {
                "Z_query_pca": np.zeros((ref.n_pcs, 0)),
                "R_query": np.zeros((ref.R_ref.shape[0], 0)),
                "Z_query_harmony": np.zeros((ref.n_pcs, 0)),
            }

        # 1) project into reference PCA
        Zq = self._project_query(X_query)             # d x m

        # 2) soft assignment to fixed reference centroids, using cosine distance
        Zq_cos = _l2_normalize_columns(Zq)
        Rq = _soft_cluster(ref.Y_cos, Zq_cos, sigma=ref.sigma)   # k x m

        # 3) query-side MoE correction using reference compression terms
        Xq_onehot, _ = _one_hot_from_batches(batch_query)
        Xq_star = np.vstack([np.ones((1, m), dtype=float), Xq_onehot])  # (1+c) x m

        d = Zq.shape[0]
        k = Rq.shape[0]
        c_plus_1 = Xq_star.shape[0]

        # final corrected query embedding
        Zq_corr = Zq.copy()

        for cluster in range(k):
            r = Rq[cluster]  # length m

            # Compute:
            # E = X* diag(r) X*^T + lambda I, with:
            #   E[0,0] += Nr[k]
            #   no ridge penalty on intercept
            # F = X* diag(r) Zq^T, with:
            #   F[0,:] += C[k,:]
            #
            # This is the compressed query-side ridge solve underlying Symphony mapping.
            XR = Xq_star * r[np.newaxis, :]    # (1+c) x m

            E = XR @ Xq_star.T                 # (1+c) x (1+c)
            E[0, 0] += ref.Nr[cluster]

            ridge = np.eye(c_plus_1, dtype=float) * ref.ridge_lambda
            ridge[0, 0] = 0.0                  # do not penalize intercept
            E = E + ridge

            F = XR @ Zq.T                      # (1+c) x d
            F[0, :] += ref.C[cluster, :]

            B = np.linalg.solve(E, F)          # (1+c) x d

            # subtract only non-intercept batch terms, weighted by cluster membership
            batch_effect = B[1:, :].T @ Xq_onehot    # d x m
            Zq_corr -= r[np.newaxis, :] * batch_effect

        return {
            "Z_query_pca": Zq,
            "R_query": Rq,
            "Z_query_harmony": Zq_corr,
        }

    def fit_transform(self, X: np.ndarray, batch: np.ndarray, treated: np.ndarray) -> Dict[str, Any]:
        """
        Convenience method:
        - build untreated reference
        - map treated cells
        - return a joint embedding in original cell order

        Returns
        -------
        dict with:
            Z_all_harmony: n_cells x d
            Z_ref_harmony: n_ref x d
            Z_query_harmony: n_query x d
            ref_mask: boolean mask
            query_mask: boolean mask
            R_query: k x n_query
        """
        self.fit(X, batch, treated)

        ref = self.ref_
        assert ref is not None

        X = _as_2d_float(X)
        batch = np.asarray(batch)
        treated = np.asarray(treated).astype(bool)

        ref_mask = ~treated
        query_mask = treated

        X_query = X[query_mask]
        batch_query = batch[query_mask]

        query_map = self._map_query(X_query, batch_query)

        Z_all = np.zeros((X.shape[0], ref.n_pcs), dtype=float)
        Z_all[ref_mask] = ref.Z_ref_harmony.T
        Z_all[query_mask] = query_map["Z_query_harmony"].T

        return {
            "Z_all_harmony": Z_all,
            "Z_ref_harmony": ref.Z_ref_harmony.T,
            "Z_query_harmony": query_map["Z_query_harmony"].T,
            "Z_query_pca": query_map["Z_query_pca"].T,
            "R_query": query_map["R_query"],
            "ref_mask": ref_mask,
            "query_mask": query_mask,
            "reference": ref,
        }

    def map_new_query(self, X_query: np.ndarray, batch_query: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Map a brand-new query onto an already fit untreated reference.
        """
        X_query = _as_2d_float(X_query)
        batch_query = np.asarray(batch_query)
        if len(batch_query) != X_query.shape[0]:
            raise ValueError("batch_query must have length n_query")
        return self._map_query(X_query, batch_query)