import numpy as np
from sklearn.linear_model import Ridge
import harmonypy as hm
import pandas as pd

# ========================
# Batch removal (Harmony)
# ========================

def _run_harmony(X_pc, slide, nclust=50, theta=None, random_state=0, verbose=False):
    """
    Robust wrapper around harmonypy.run_harmony.

    Input:
      X_pc: (n_cells, n_pcs)
      slide: (n_cells,)
    Output:
      X_corr: (n_cells, n_pcs)
    """
    X_pc = np.asarray(X_pc, dtype=float)
    slide = np.asarray(slide)

    n_cells, n_pcs = X_pc.shape

    meta = pd.DataFrame({"slide": pd.Categorical(slide.astype(str))})

    ho = hm.run_harmony(
        X_pc.T,                  # harmonypy typically expects (features, cells)
        meta,
        vars_use=["slide"],
        nclust=nclust,
        theta=theta,
        random_state=random_state,
        verbose=verbose
    )

    Z = np.asarray(ho.Z_corr)

    # Harmonypy variants may return (n_pcs, n_cells) or (n_cells, n_pcs).
    if Z.shape == (n_pcs, n_cells):
        X_corr = Z.T
    elif Z.shape == (n_cells, n_pcs):
        X_corr = Z
    elif Z.shape[0] == n_pcs and Z.shape[1] != n_cells:
        # best-effort: assume features-first, transpose
        X_corr = Z.T
    elif Z.shape[1] == n_pcs and Z.shape[0] != n_cells:
        # best-effort: assume cells-first
        X_corr = Z
    else:
        raise ValueError(
            f"Unexpected Harmony output shape {Z.shape}; expected {(n_pcs, n_cells)} or {(n_cells, n_pcs)}."
        )

    # Final sanity check
    if X_corr.shape != (n_cells, n_pcs):
        raise ValueError(
            f"Harmony corrected shape {X_corr.shape} does not match input {(n_cells, n_pcs)}."
        )

    return X_corr

def harmony(
    X_pca, slide_id, # treated,
    *,
    # cell state
    d_state=30,
    harmony_nclust=50,
    harmony_theta=None,
    center_kwargs = None,
):
    X = np.asarray(X_pca, dtype=float)
    slide_id = np.asarray(slide_id)
    # treated = np.asarray(treated).astype(bool)
    n, d = X.shape
    
    if center_kwargs is None:
        center_kwargs = {}
        
    # X = center_slide_untreated_fps(X, slide_id, treated,**center_kwargs)
    # Xs, mu, sd = _center_scale(X)
    
    X_h_all = _run_harmony(X, slide_id, nclust=harmony_nclust, theta=harmony_theta)
    
    return X_h_all

def center_slide_untreated_mean(X, slide_id, treated, eps=1e-8):
    X = np.asarray(X, float).copy()
    slide_id = np.asarray(slide_id)
    treated = np.asarray(treated).astype(bool)
    untreated = ~treated

    global_mean = X[untreated].mean(axis=0, keepdims=True)

    for s in np.unique(slide_id):
        idx_u = (slide_id == s) & untreated
        if idx_u.sum() < 5:
            # fallback: use all cells in slide if too few untreated (rare)
            idx_u = (slide_id == s)
        mu_s = X[idx_u].mean(axis=0, keepdims=True)
        X[slide_id == s] = X[slide_id == s] - mu_s + global_mean

    return X

def _center_scale(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    # sd = 1
    return (X - mu) / sd, mu, sd

def _pairwise_sq_dists(A, B=None):
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    aa = np.sum(A * A, axis=1, keepdims=True)
    bb = np.sum(B * B, axis=1, keepdims=True).T
    D2 = aa + bb - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0)


def _fps_indices(X, n_landmarks, start="medoid", rng=None):
    """
    Farthest point sampling indices from X.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n_landmarks >= n:
        return np.arange(n)

    if rng is None:
        rng = np.random.default_rng(0)

    # choose first point
    if start == "random":
        first = rng.integers(n)
    elif start == "medoid":
        # exact medoid initialization
        D2 = _pairwise_sq_dists(X)
        first = np.argmin(D2.sum(axis=1))
    else:
        raise ValueError("start must be 'medoid' or 'random'")

    selected = [first]
    min_d2 = _pairwise_sq_dists(X, X[[first]]).ravel()

    for _ in range(1, n_landmarks):
        nxt = np.argmax(min_d2)
        selected.append(nxt)
        d2_new = _pairwise_sq_dists(X, X[[nxt]]).ravel()
        min_d2 = np.minimum(min_d2, d2_new)

    return np.array(selected, dtype=int)


def _landmark_center(X_landmarks, method="medoid"):
    """
    Compute center from landmark set.
    """
    X_landmarks = np.asarray(X_landmarks, dtype=float)

    if method == "mean":
        return X_landmarks.mean(axis=0, keepdims=True)

    if method == "medoid":
        D2 = _pairwise_sq_dists(X_landmarks)
        idx = np.argmin(D2.sum(axis=1))
        return X_landmarks[[idx]]

    raise ValueError("method must be 'mean' or 'medoid'")


def center_slide_untreated_fps(
    X,
    slide_id,
    treated,
    n_landmarks=50,
    center_method="medoid",
    fps_start="medoid",
    min_untreated=5,
    use_global_untreated_only=True,
    random_state=0,
    return_centers=False,
):
    """
    Center each slide using a shape-based center computed from FPS landmarks.

    Parameters
    ----------
    X : array, shape (n, d)
        Feature matrix.
    slide_id : array, shape (n,)
        Slide label per row.
    treated : array, shape (n,)
        Boolean or 0/1 treatment indicator.
    n_landmarks : int
        Number of FPS landmarks per slide.
    center_method : {'mean', 'medoid'}
        How to define center from the FPS landmarks.
    fps_start : {'medoid', 'random'}
        Initialization for FPS.
    min_untreated : int
        If a slide has fewer untreated cells than this, fall back to all cells in that slide.
    use_global_untreated_only : bool
        If True, global target center is built from untreated cells only.
        If False, global target center uses all cells.
    random_state : int
        Seed for reproducibility.
    return_centers : bool
        If True, also return per-slide and global centers.

    Returns
    -------
    X_centered : array, shape (n, d)
        Centered matrix.
    info : dict, optional
        Returned only if return_centers=True.
    """
    X = np.asarray(X, dtype=float).copy()
    slide_id = np.asarray(slide_id)
    treated = np.asarray(treated).astype(bool)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if slide_id.ndim != 1 or slide_id.shape[0] != X.shape[0]:
        raise ValueError("slide_id must be 1D with same length as X.")
    if treated.ndim != 1 or treated.shape[0] != X.shape[0]:
        raise ValueError("treated must be 1D with same length as X.")
    if n_landmarks <= 0:
        raise ValueError("n_landmarks must be positive.")

    untreated = ~treated
    rng = np.random.default_rng(random_state)

    # ----- global reference pool -----
    if use_global_untreated_only and untreated.sum() >= min_untreated:
        global_pool = X[untreated]
    else:
        global_pool = X

    global_idx = _fps_indices(
        global_pool,
        n_landmarks=min(n_landmarks, global_pool.shape[0]),
        start=fps_start,
        rng=rng,
    )
    global_landmarks = global_pool[global_idx]
    global_center = _landmark_center(global_landmarks, method=center_method)

    # ----- per-slide alignment -----
    X_out = X.copy()
    slides = np.unique(slide_id)
    slide_centers = {}

    for s in slides:
        idx_slide = (slide_id == s)
        idx_u = idx_slide & untreated

        if idx_u.sum() >= min_untreated:
            pool = X[idx_u]
            used = "untreated"
        else:
            pool = X[idx_slide]
            used = "all"

        m = min(n_landmarks, pool.shape[0])
        fps_idx = _fps_indices(pool, n_landmarks=m, start=fps_start, rng=rng)
        landmarks = pool[fps_idx]
        center_s = _landmark_center(landmarks, method=center_method)

        X_out[idx_slide] = X_out[idx_slide] - center_s + global_center
        slide_centers[s] = {
            "center": center_s.ravel(),
            "n_pool": pool.shape[0],
            "pool_used": used,
            "n_landmarks": m,
        }

    if return_centers:
        return X_out, {
            "global_center": global_center.ravel(),
            "slide_centers": slide_centers,
        }

    return X_out