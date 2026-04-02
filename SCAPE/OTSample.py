import numpy as np

def sample_map_projection(
    mapping,
    X_source,
    n_samples=10,
    with_replacement=True,
    return_indices=False,
    random_state=None,
):
    """
    Sample projection(s) from a row-stochastic mapping matrix.

    Parameters
    ----------
    mapping : array-like, shape (n_target, n_source)
        Mapping matrix from target cells to source cells.
        Each row corresponds to one target cell and contains weights over source cells.
        Rows do not need to be perfectly normalized; they will be renormalized internally.

    X_source : array-like, shape (n_source, d)
        Source embedding coordinates or features.

    n_samples : int, default=10
        Number of sampled projected embeddings to generate.

    with_replacement : bool, default=True
        If True, each target cell is sampled independently in each replicate.
        If False, sampling is done without replacement within each row across replicates
        when possible. Usually True is what you want.

    return_indices : bool, default=False
        If True, also return the sampled source indices of shape (n_samples, n_target).

    random_state : int or np.random.Generator or None
        Random seed or Generator for reproducibility.

    Returns
    -------
    X_proj_samples : ndarray, shape (n_samples, n_target, d)
        Sampled projected embeddings.

    sampled_idx : ndarray, shape (n_samples, n_target), optional
        Sampled source index for each target cell in each replicate.
        Only returned if return_indices=True.
    """
    mapping = np.asarray(mapping, dtype=np.float64)
    X_source = np.asarray(X_source, dtype=np.float64)

    if mapping.ndim != 2:
        raise ValueError("mapping must be 2D, shape (n_target, n_source)")
    if X_source.ndim != 2:
        raise ValueError("X_source must be 2D, shape (n_source, d)")
    if mapping.shape[1] != X_source.shape[0]:
        raise ValueError(
            f"mapping.shape[1] ({mapping.shape[1]}) must equal X_source.shape[0] ({X_source.shape[0]})"
        )
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    n_target, n_source = mapping.shape
    d = X_source.shape[1]

    # Clean numerical issues
    W = mapping.copy()
    W[~np.isfinite(W)] = 0.0
    W[W < 0] = 0.0

    row_sums = W.sum(axis=1, keepdims=True)

    # For rows with zero total mass, fall back to uniform over all source cells
    zero_rows = (row_sums[:, 0] <= 0)
    if np.any(zero_rows):
        W[zero_rows, :] = 1.0
        row_sums = W.sum(axis=1, keepdims=True)

    W = W / row_sums

    sampled_idx = np.empty((n_samples, n_target), dtype=np.int64)

    if with_replacement:
        for j in range(n_target):
            sampled_idx[:, j] = rng.choice(
                n_source,
                size=n_samples,
                replace=True,
                p=W[j],
            )
    else:
        for j in range(n_target):
            support = np.flatnonzero(W[j] > 0)
            probs = W[j, support]

            if support.size == 0:
                support = np.arange(n_source)
                probs = np.ones(n_source, dtype=np.float64) / n_source
            else:
                probs = probs / probs.sum()

            if n_samples <= support.size:
                chosen = rng.choice(
                    support,
                    size=n_samples,
                    replace=False,
                    p=probs,
                )
            else:
                # Not enough support points, so fall back to replacement
                chosen = rng.choice(
                    support,
                    size=n_samples,
                    replace=True,
                    p=probs,
                )

            sampled_idx[:, j] = chosen

    X_proj_samples = X_source[sampled_idx]  # shape (n_samples, n_target, d)

    if return_indices:
        return X_proj_samples, sampled_idx
    return X_proj_samples