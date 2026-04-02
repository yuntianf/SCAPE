from __future__ import annotations

import numpy as np
from typing import Callable, Optional


def simulate_observed_confounder(
    n_slides_pre: int,
    n_slides_post: int,
    cell_n = 1000,
    d_u: int = 8,
    d_c: int = 6,

    # U distribution (same pre/post)
    U_simulator: Optional[Callable] = None, #[continuity_generator,cluster_generator]
    U_simulator_kwargs=None,

    # Treatment assignment: observational confounding
    w_t_mu = 1,
    w_t_sd: float = 0.6,
    t_intercept: float = -0.2,
    t_noise_sd: float = 0.0,   # extra noise in treatment logit

    # Outcome signal
    w_yu_mu = 1,
    w_yu_sd: float = 0.7,
    tau: float = 1.0,
    y_noise_sd: float = 0.2,

    # Collider model: ALWAYS has Y->C; post adds T->C
    alpha_y: float = 1.0,       # Y -> C (all slides)
    alpha_t: float = 1.0,       # T -> C (post slides only)
    c_noise_sd: float = 0.2,

    # --- Batch effect controls (slide-level) ---
    add_batch: bool = True,
    batch_additive_sd: float = 0.8,
    batch_rank: int = 3,             # 0 => i.i.d per-feature shift; >0 low-rank
    batch_scale_sd: float = 0.0,     # multiplicative log-scale per slide; 0 disables
    batch_diff_pre_post: bool = False,
    batch_post_multiplier: float = 1.0,
    batch_rot_sd=0.05,
    batch_mix_rank=2,
    batch_mix_sd=0.08,

    standardize: bool = False,
    seed: int = 0
):
    """
    Simulate observed confounder matrix K = [U | C_obs] with:
      - U: true confounders (same distribution in pre/post)
      - T: confounded treatment depending on U
      - Y: depends on T and U (plus noise)
      - C_obs: collider-like features:
          * all slides:      C = alpha_y * Y + eps_C
          * post slides:     C += alpha_t * T  (optionally gated by T)
      - optional slide-level batch effect applied to observed features

    Returns a dict with K, U, C, T, Y, slide_id, is_post, plus batch parameters used.
    """

    rng = np.random.default_rng(seed)
    n_slides_total = n_slides_pre + n_slides_post
    n_cells_per_slide = rng.poisson(lam=cell_n, size=n_slides_total)

    # cells per slide
    if isinstance(n_cells_per_slide, int):
        n_cells = [n_cells_per_slide] * n_slides_total
    else:
        n_cells = list(n_cells_per_slide)
        if len(n_cells) != n_slides_total:
            raise ValueError("n_cells_per_slide must have length n_slides_pre + n_slides_post")

    d_obs = d_u + d_c

    # sample weights
    w_t = rng.normal(w_t_mu, w_t_sd, size=(d_u,))
    w_yu = rng.normal(w_yu_mu, w_yu_sd, size=(d_u,))

    # ---------------- confounder simulator ----------------
    if U_simulator is None:
        U_simulator = continuity_generator
    
    if U_simulator_kwargs is None:
        U_simulator_kwargs = {}
    
    if U_simulator is continuity_generator:
        U_sampler = continuity_sampler
    elif U_simulator is cluster_generator:
        U_sampler = cluster_sampler
    else:
        raise ValueError("Unknown U_simulator; please also specify the matching sampler.")
    
    # ---------------- simulate ----------------
    rows_K, rows_X, rows_U, rows_C = [], [], [], []
    row_p = []
    rows_T, rows_Y = [], []
    rows_slide, rows_is_post, rows_cellid = [], [], []
    
    rows_S = []

    U_pool = U_simulator(rng, d_u=d_u, **U_simulator_kwargs)
    
    for s in range(n_slides_total):
        is_post = (s >= n_slides_pre)
        n = n_cells[s]

        # U
        U, S = U_sampler(rng, n, U_pool)

        # T | U
        logits = t_intercept + center_scale(U) @ w_t
        if t_noise_sd > 0:
            logits = logits + rng.normal(0, t_noise_sd, size=n)
        logits = 2*logits/(logits.std())
        p = sigmoid(logits)
        # p = S["t"]*0.5+0.1
        T = rng.binomial(1, p, size=n).astype(float)
        

        if not is_post:
            T = np.zeros(n)
            
        # Y | U,T
        Y_raw = (U @ w_yu) + rng.normal(0, y_noise_sd, size=n)
        Y_raw = center_scale(Y_raw)
        Y = Y_raw + tau * T 

        # C exists in ALL slides via Y->C
        # (replicate Y across d_c dims + feature noise)
        C_noise = rng.normal(0, c_noise_sd, size=(n, d_c))
        
        C_raw = alpha_y * Y_raw[:, None]  + C_noise
        C = alpha_y * Y[:, None] + C_noise
        
        # Add T->C ONLY in post slides (optionally gated)
        if is_post and alpha_t != 0:
            C = C + alpha_t * T[:, None]

        C_obs = C  # observed collider block
        
        # The coordinates if C is not influenced by T
        X = np.concatenate([U, C_raw], axis=1)
        
        # observed K (before batch)
        K_clean = np.concatenate([U, C_obs], axis=1)

        # collect
        rows_K.append(K_clean)
        rows_X.append(X)
        rows_U.append(U)
        rows_C.append(C)  # the collider values used in C_obs
        rows_T.append(T)
        rows_Y.append(Y)
        row_p.append(p)
        rows_slide.append(np.full(n, s, dtype=int))
        rows_is_post.append(np.full(n, is_post, dtype=bool))
        rows_cellid.append(np.arange(n, dtype=int))
        
        rows_S.append(S["t"])

    K = np.vstack(rows_K)
    X = np.vstack(rows_X)
    U = np.vstack(rows_U)
    C = np.vstack(rows_C)
    T = np.concatenate(rows_T)
    Y = np.concatenate(rows_Y)
    p = np.concatenate(row_p)
    slide_id = np.concatenate(rows_slide)
    is_post = np.concatenate(rows_is_post)
    cell_id_within_slide = np.concatenate(rows_cellid)
    
    S = np.concatenate(rows_S)

    if standardize:
        mu = K.mean(axis=0, keepdims=True)
        sd = K.std(axis=0, keepdims=True) + 1e-8
        K = (K - mu) / sd

    if add_batch:
        K = apply_batch_all(
            K,
            slide_id,
            batch_additive_sd=batch_additive_sd,
            batch_rank=batch_rank,
            batch_scale_sd=batch_scale_sd,
            batch_diff_pre_post=batch_diff_pre_post,
            is_post=is_post,
            batch_post_multiplier=batch_post_multiplier,
            batch_rot_sd=batch_rot_sd,
            batch_mix_rank=batch_mix_rank,
            batch_mix_sd=batch_mix_sd,
            seed=seed + 12345 if seed is not None else None,
        )
        
    return {
        "K": K,
        "X": X,
        "U": U,
        "C": C,
        "T": T,
        "Y": Y,
        "p": p,
        "slide_id": slide_id,
        "is_post": is_post,
        "cell_id_within_slide": cell_id_within_slide,
        "feature_slices": {"U": slice(0, d_u), "C": slice(d_u, d_u + d_c)},
        "params": {
            "n_slides_pre": n_slides_pre,
            "n_slides_post": n_slides_post,
            "n_cells_per_slide": n_cells,
            "d_u": d_u,
            "d_c": d_c,
            "alpha_y": alpha_y,
            "alpha_t": alpha_t,
            "batch_additive_sd": batch_additive_sd,
            "batch_rank": batch_rank,
            "batch_scale_sd": batch_scale_sd,
            "batch_diff_pre_post": batch_diff_pre_post,
            "batch_post_multiplier": batch_post_multiplier,
        },
        
        "S":S
    }

# =========================================
# Ground truth confounding matrix generator
# =========================================
def continuity_generator(
    rng,
    d_u: int,
    n_programs: int = 6,
    n_branches: int = 3,
    load_scale: float = 2.0,
    prog_amp: float = 1.0,
    prog_width: float = 0.08,     # smaller = sharper transitions
    shared_frac: float = 0.5,     # fraction of programs shared across branches
    noise_sd: float = 0.3,
    noise_rank: int = 3,
    noise_corr_sd: float = 0.4
):
    """
    Create fixed parameters for a biological manifold generator.

    Programs are sigmoids along pseudotime; some are branch-specific.
    U = G(t,b) @ W + correlated_noise
    """
    # program -> feature loadings
    W = rng.normal(0, 1.0, size=(n_programs, d_u))
    W = W / (np.sqrt((W**2).mean()) + 1e-8) * load_scale

    # program kinetics: each program has an "activation time" and a direction (+/-)
    t0 = rng.uniform(0.15, 0.85, size=(n_programs,))
    direction = rng.choice([-1.0, 1.0], size=(n_programs,))
    amp = prog_amp * rng.uniform(0.6, 1.4, size=(n_programs,))

    # branch specificity mask (continuous biology-like divergence)
    n_shared = int(np.round(shared_frac * n_programs))
    shared_idx = np.arange(n_shared)
    branch_idx = np.arange(n_shared, n_programs)

    # branch weights for branch-specific programs
    # for shared programs, all branches weight = 1
    branch_weight = np.ones((n_branches, n_programs))
    if len(branch_idx) > 0:
        # make branch-specific programs have different amplitudes per branch
        branch_weight[:, branch_idx] = rng.uniform(0.0, 1.5, size=(n_branches, len(branch_idx)))

    # correlated noise structure (fixed across slides)
    if noise_rank > 0 and noise_corr_sd > 0:
        B = rng.normal(0, 1.0, size=(d_u, noise_rank))
        B = B / (np.sqrt((B**2).mean()) + 1e-8) * noise_corr_sd
    else:
        B = None

    return {
        "W": W, "t0": t0, "direction": direction, "amp": amp,
        "prog_width": prog_width,
        "branch_weight": branch_weight,
        "n_branches": n_branches,
        "noise_sd": noise_sd, "B": B
    }

def continuity_sampler(
    rng,
    n: int,
    gen: dict,
    branch_probs=None,
    t_dist="beta",          # "uniform" or "beta"
    beta_a=2.0,
    beta_b=2.0
):
    """
    Sample a biological manifold:
      - branch b
      - pseudotime t
      - program activities G(t,b) via sigmoids
      - embed to U
    """
    n_branches = gen["n_branches"]
    if branch_probs is None:
        branch_probs = np.ones(n_branches) / n_branches
    else:
        branch_probs = np.asarray(branch_probs, float)
        branch_probs = branch_probs / branch_probs.sum()

    b = rng.choice(n_branches, size=n, p=branch_probs)

    if t_dist == "uniform":
        t = rng.uniform(0, 1, size=n)
    elif t_dist == "beta":
        t = rng.beta(beta_a, beta_b, size=n)
    else:
        raise ValueError("t_dist must be 'uniform' or 'beta'")

    # sigmoid programs along pseudotime
    # g_j(t) = amp_j * dir_j * sigmoid((t - t0_j)/width)
    width = gen["prog_width"]
    z = (t[:, None] - gen["t0"][None, :]) / max(width, 1e-6)
    sig = 1.0 / (1.0 + np.exp(-z))
    G = gen["amp"][None, :] * gen["direction"][None, :] * sig

    # apply branch-specific weights
    G = G * gen["branch_weight"][b, :]

    # embed to U
    U = G @ gen["W"]

    # add correlated + isotropic noise
    U = U + rng.normal(0, gen["noise_sd"], size=U.shape)
    if gen["B"] is not None:
        zc = rng.normal(0, 1.0, size=(n, gen["B"].shape[1]))
        U = U + zc @ gen["B"].T

    return U, {"t": t, "branch": b, "G": G}


def cluster_generator(
    rng,
    d_u: int,
    n_clusters: int = 3,
    q_intra: int = 1,              # intrinsic dimension inside each cluster: 1=trajectory, 2=sheet
    curve_type: str = "spline",    # {"spline","circle","sine"}
    cluster_sep: float = 6.0,      # distance between cluster centers
    intra_scale: float = 2.5,      # size of within-cluster manifold
    noise_sd: float = 0.4,         # thickness
    # optional correlated noise shared across all clusters
    noise_rank: int = 3,
    noise_corr_sd: float = 0.3,
):
    """
    Creates global parameters for a mixture of continuous manifolds.
    Each cluster k has:
      - center mu_k
      - a smooth curve/sheet f_k(t) embedded in R^{d_u}
    """
    # cluster centers
    mu = rng.normal(0, 1.0, size=(n_clusters, d_u))
    mu = mu / (np.sqrt((mu**2).mean()) + 1e-8) * cluster_sep

    # per-cluster manifold basis directions (orth-ish)
    # B_k: (q_intra, d_u) directions for manifold coordinates
    B = rng.normal(0, 1.0, size=(n_clusters, q_intra, d_u))
    B = B / (np.sqrt((B**2).mean()) + 1e-8) * intra_scale

    # curve shape parameters per cluster
    if curve_type == "spline":
        # cubic spline-like: sum_j a_j * phi_j(t) in each intra dim
        # We'll implement as low-order polynomial features for simplicity.
        # Each cluster has coefficients for intra dims: coef[k, intra_dim, poly_deg+1]
        poly_deg = 3
        coef = rng.normal(0, 1.0, size=(n_clusters, q_intra, poly_deg + 1))
        coef = coef / (np.sqrt((coef**2).mean()) + 1e-8)
        shape = {"poly_deg": poly_deg, "coef": coef}
    elif curve_type == "circle":
        # needs q_intra >= 2 to form a loop
        if q_intra < 2:
            raise ValueError("curve_type='circle' requires q_intra >= 2")
        phase = rng.uniform(0, 2*np.pi, size=(n_clusters,))
        shape = {"phase": phase}
    elif curve_type == "sine":
        freq = rng.uniform(0.5, 2.0, size=(n_clusters, q_intra))
        phase = rng.uniform(0, 2*np.pi, size=(n_clusters, q_intra))
        shape = {"freq": freq, "phase": phase}
    else:
        raise ValueError("curve_type must be one of {'spline','circle','sine'}")

    # global correlated noise
    if noise_rank > 0 and noise_corr_sd > 0:
        L = rng.normal(0, 1.0, size=(d_u, noise_rank))
        L = L / (np.sqrt((L**2).mean()) + 1e-8) * noise_corr_sd
    else:
        L = None

    return {
        "d_u": d_u,
        "n_clusters": n_clusters,
        "q_intra": q_intra,
        "curve_type": curve_type,
        "mu": mu,
        "B": B,
        "shape": shape,
        "noise_sd": noise_sd,
        "L": L,
    }


def _poly_features(t, deg):
    # t: (n,)
    # returns (n, deg+1): [1, t, t^2, ..., t^deg]
    feats = [np.ones_like(t)]
    for p in range(1, deg + 1):
        feats.append(t**p)
    return np.stack(feats, axis=1)


def cluster_sampler(
    rng,
    n: int,
    gen: dict,
    cluster_probs=None,
    t_dist="uniform",     # {"uniform","beta"}
    beta_a=2.0,
    beta_b=2.0,
    allow_clusterwise_t_dist=False,  # if True, each cluster gets its own beta params
):
    """
    Samples U from a cluster-wise manifold mixture with shared global params.
    Returns U and latent labels (cluster k and continuous t).
    """
    K = gen["n_clusters"]
    d_u = gen["d_u"]
    q = gen["q_intra"]

    if cluster_probs is None:
        cluster_probs = np.ones(K) / K
    else:
        cluster_probs = np.asarray(cluster_probs, float)
        cluster_probs = cluster_probs / cluster_probs.sum()

    k = rng.choice(K, size=n, p=cluster_probs)

    # sample continuous intra-cluster coordinate(s)
    if q == 1:
        if t_dist == "uniform":
            t = rng.uniform(0, 1, size=n)
        elif t_dist == "beta":
            if allow_clusterwise_t_dist:
                # each cluster has different density along the trajectory (still same across slides if probs fixed)
                a_k = rng.uniform(1.2, 3.0, size=K)
                b_k = rng.uniform(1.2, 3.0, size=K)
                t = np.empty(n)
                for kk in range(K):
                    idx = (k == kk)
                    t[idx] = rng.beta(a_k[kk], b_k[kk], size=idx.sum())
            else:
                t = rng.beta(beta_a, beta_b, size=n)
        else:
            raise ValueError("t_dist must be 'uniform' or 'beta'")
        T = t[:, None]  # (n,1)
    else:
        # sheet: sample (t1,t2)
        if t_dist == "uniform":
            T = rng.uniform(0, 1, size=(n, q))
        elif t_dist == "beta":
            T = rng.beta(beta_a, beta_b, size=(n, q))
        else:
            raise ValueError("t_dist must be 'uniform' or 'beta'")
        t = T  # (n,q)

    # build intra-manifold coordinates F in R^{q}
    curve_type = gen["curve_type"]
    if curve_type == "spline":
        # F_k(t) is polynomial in each intra dim
        deg = gen["shape"]["poly_deg"]
        feats = _poly_features(T[:, 0] if q >= 1 else T.squeeze(), deg)  # (n,deg+1)
        # For q>1, reuse same t0 for simplicity; you can expand if you want
        F = np.zeros((n, q))
        coef = gen["shape"]["coef"]  # (K,q,deg+1)
        for kk in range(K):
            idx = (k == kk)
            if idx.sum() == 0:
                continue
            for qi in range(q):
                F[idx, qi] = feats[idx] @ coef[kk, qi]
        # normalize F to roughly [-1,1]
        F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-8)

    elif curve_type == "circle":
        # use first two dims as circle, remaining dims (if any) as smooth functions of t
        phase = gen["shape"]["phase"]
        angle = 2*np.pi*(T[:, 0])  # (n,)
        F = np.zeros((n, q))
        for kk in range(K):
            idx = (k == kk)
            if idx.sum() == 0:
                continue
            ang = angle[idx] + phase[kk]
            F[idx, 0] = np.cos(ang)
            F[idx, 1] = np.sin(ang)
            # extra dims (if q>2): gentle sine of angle
            for qi in range(2, q):
                F[idx, qi] = np.sin((qi+1)*ang) / (qi+1)

    elif curve_type == "sine":
        freq = gen["shape"]["freq"]
        phase = gen["shape"]["phase"]
        F = np.zeros((n, q))
        for kk in range(K):
            idx = (k == kk)
            if idx.sum() == 0:
                continue
            for qi in range(q):
                F[idx, qi] = np.sin(2*np.pi*freq[kk, qi]*T[idx, qi] + phase[kk, qi])
    else:
        raise ValueError("unsupported curve_type")

    # embed into d_u: U = mu_k + F @ B_k + noise
    U = np.zeros((n, d_u))
    for kk in range(K):
        idx = (k == kk)
        if idx.sum() == 0:
            continue
        U[idx] = gen["mu"][kk] + F[idx] @ gen["B"][kk]

    # add noise
    U = U + rng.normal(0, gen["noise_sd"], size=U.shape)
    if gen["L"] is not None:
        z = rng.normal(0, 1.0, size=(n, gen["L"].shape[1]))
        U = U + z @ gen["L"].T

    latent = {"t": k, "distirbution": t, "F": F}
    return U, latent

def make_random_orthogonal(rng, d):
    # random orthogonal matrix via QR
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    # make determinant +1 (proper rotation)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q

def make_near_identity_rotation(rng, d, rot_sd=0.05):
    """
    Small rotation around identity.
    Uses a random skew-symmetric matrix and matrix exponential approximation.
    """
    A = rng.normal(0, rot_sd, size=(d, d))
    S = A - A.T                       # skew-symmetric
    # first-order exp(S) ≈ I + S is okay for small rot_sd
    R = np.eye(d) + S
    return R

def apply_batch_all(
    K,
    slide_id,
    *,
    batch_additive_sd: float = 0.8,
    batch_rank: int = 3,
    batch_scale_sd: float = 0.0,
    batch_diff_pre_post: bool = False,
    is_post=None,
    batch_post_multiplier: float = 1.0,
    batch_rot_sd: float = 0.05,
    batch_mix_rank: int = 2,
    batch_mix_sd: float = 0.08,
    seed: int | None = None,
    return_params: bool = False,
):
    """
    Apply slide-level batch effects to a full matrix K.

    Parameters
    ----------
    K : array-like, shape (n, d)
        Full data matrix containing all cells from all slides.
    slide_id : array-like, shape (n,)
        Integer or hashable slide label for each row of K.
    batch_additive_sd : float
        Magnitude of additive per-slide shifts.
    batch_rank : int
        Rank of low-rank additive structure across slides.
        If 0, use iid per-feature additive shifts.
    batch_scale_sd : float
        SD of per-slide log-scale effects.
    batch_diff_pre_post : bool
        Whether to amplify batch effects on post slides.
    is_post : array-like, shape (n,), optional
        Boolean vector indicating whether each row is from a post slide.
        Required if batch_diff_pre_post=True.
    batch_post_multiplier : float
        Multiplier applied to post-slide additive and scale effects.
    batch_rot_sd : float
        Strength of near-identity rotation per slide.
    batch_mix_rank : int
        Rank of low-rank feature-mixing matrix per slide.
    batch_mix_sd : float
        Magnitude of low-rank feature mixing.
    seed : int or None
        Random seed.
    return_params : bool
        If True, also return batch parameters.

    Returns
    -------
    K_obs : ndarray, shape (n, d)
        Batched data matrix.
    params : dict, optional
        Slide-level batch parameters if return_params=True.
    """
    rng = np.random.default_rng(seed)

    K = np.asarray(K, dtype=float)
    slide_id = np.asarray(slide_id)

    if K.ndim != 2:
        raise ValueError("K must be 2D.")
    if slide_id.ndim != 1 or slide_id.shape[0] != K.shape[0]:
        raise ValueError("slide_id must be 1D with len(slide_id) == K.shape[0].")

    n, d = K.shape
    slides, inv = np.unique(slide_id, return_inverse=True)
    n_slides = len(slides)

    # ---- additive batch effect ----
    if batch_rank > 0:
        L = rng.normal(0.0, 1.0, size=(batch_rank, d))
        L = L / (np.sqrt((L ** 2).mean()) + 1e-8) * batch_additive_sd
        Z = rng.normal(0.0, 1.0, size=(n_slides, batch_rank))
        b_all = Z @ L
    else:
        b_all = rng.normal(0.0, batch_additive_sd, size=(n_slides, d))

    # ---- multiplicative log-scale effect ----
    if batch_scale_sd > 0:
        a_all = rng.normal(0.0, batch_scale_sd, size=(n_slides, d))
    else:
        a_all = np.zeros((n_slides, d))

    # ---- optional pre/post amplification ----
    if batch_diff_pre_post:
        if is_post is None:
            raise ValueError("is_post must be provided when batch_diff_pre_post=True.")
        is_post = np.asarray(is_post).astype(bool)
        if is_post.shape[0] != n:
            raise ValueError("is_post must have length n.")
        # infer slide-level post flag
        slide_is_post = np.zeros(n_slides, dtype=bool)
        for s in range(n_slides):
            vals = is_post[inv == s]
            if len(vals) == 0:
                continue
            # require consistency within slide
            if np.any(vals != vals[0]):
                raise ValueError("Rows within a slide have inconsistent is_post values.")
            slide_is_post[s] = vals[0]

        b_all[slide_is_post] *= batch_post_multiplier
        a_all[slide_is_post] *= batch_post_multiplier

    # ---- per-slide linear transforms ----
    A_all = np.zeros((n_slides, d, d), dtype=float)
    for s in range(n_slides):
        R = make_near_identity_rotation(rng, d, rot_sd=batch_rot_sd)

        if batch_mix_rank > 0 and batch_mix_sd > 0:
            U = rng.normal(0.0, 1.0, size=(d, batch_mix_rank))
            V = rng.normal(0.0, 1.0, size=(d, batch_mix_rank))
            M = (U @ V.T) / np.sqrt(d) * batch_mix_sd
        else:
            M = np.zeros((d, d))

        A_all[s] = R + M

    # ---- apply transform slide by slide ----
    K_obs = np.empty_like(K, dtype=float)
    for s in range(n_slides):
        idx = (inv == s)
        Xs = K[idx] @ A_all[s]
        Xs = Xs * np.exp(a_all[s]) + b_all[s]
        K_obs[idx] = Xs

    if return_params:
        return K_obs, {
            "slides": slides,
            "b_all": b_all,
            "a_all": a_all,
            "A_all": A_all,
        }
    return K_obs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))\

def center_scale(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    
    return (X - mu) / sd