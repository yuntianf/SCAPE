from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph, NearestNeighbors
from scipy.sparse import csr_matrix, diags
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Tuple, Dict, Union


def neighbor_type_counts(
    coords: Union[np.ndarray, pd.DataFrame],
    cell_types: Iterable,
    mode: str = "radius",               # "radius" or "knn"
    radius: Optional[float] = None,     # used if mode="radius"
    k: Optional[int] = None,            # used if mode="knn"
    *,
    # --- NEW: multi-slide support ---
    slide_ids: Optional[Iterable] = None,     # length n; restricts neighbors within slide if provided
    same_slide: bool = True,                  # only search within the same slide
    radius_map: Optional[Dict[object, float]] = None,  # per-slide radius overrides (mode="radius")
    k_map: Optional[Dict[object, int]] = None,         # per-slide k overrides (mode="knn")
    # --------------------------------
    target_mask: Optional[Iterable] = None,   # optional bool array (n,) selecting target cells
    target_types: Optional[Union[Iterable, object]] = None,  # scalar or list of types to target
    include_self: bool = False,         # usually False
    metric: str = "euclidean",          # passed to sklearn
    ids: Optional[Iterable] = None      # optional array-like of length n for nicer row index
) -> pd.DataFrame:
    """
    Return a DataFrame of neighbor-type counts per target cell, with optional
    slide-aware neighbor search (block-diagonal adjacency by slide).

    Parameters
    ----------
    coords : (n, d) array-like or DataFrame
        Cell coordinates (pixels or microns). d = 2 or 3.
    cell_types : (n,) array-like
        Cell type labels (str/int). Order corresponds to coords.
    mode : {"radius","knn"}
        Neighborhood definition.
    radius : float
        Global search radius when mode="radius".
    k : int
        Global # of neighbors when mode="knn".
    slide_ids : (n,) array-like, optional
        Slide label per cell. If provided and same_slide=True, neighbors are searched
        only within each slide. If same_slide=False, slides are ignored.
    same_slide : bool
        Whether to restrict neighbor search to the same slide (default True).
    radius_map : dict, optional
        Per-slide radius overrides when mode="radius", e.g. {slideA: 30.0, slideB: 25.0}.
        If a slide key is missing, falls back to `radius`.
    k_map : dict, optional
        Per-slide k overrides when mode="knn", e.g. {slideA: 30, slideB: 20}.
        If a slide key is missing, falls back to `k`.
    target_mask : (n,) bool array, optional
        If provided, only these cells are treated as targets.
    target_types : scalar or list, optional
        Alternative to target_mask: keep targets whose type is in this set.
    include_self : bool
        Whether to count the cell itself as a neighbor.
    metric : str
        Distance metric for sklearn neighbor graph.
    ids : array-like, optional
        Custom row index (e.g., cell IDs). Must be length n.

    Returns
    -------
    df : pandas.DataFrame, shape (n_targets, m_types)
        Rows = target cells; columns = unique cell types; values = counts.
        Row index = `ids[target_indices]` if ids provided, else original indices.
    """
    # --- inputs / alignment ---
    coords = np.asarray(coords)
    n = coords.shape[0]
    cell_types = np.asarray(cell_types)
    if cell_types.shape[0] != n:
        raise ValueError("cell_types must have length n (number of rows in coords).")
    if ids is not None and len(ids) != n:
        raise ValueError("`ids` must have length n.")

    # map types to integer codes with stable ordering
    types, type_codes = np.unique(cell_types, return_inverse=True)
    m = len(types)

    # targets
    if target_mask is None:
        if target_types is None:
            target_indices = np.arange(n)
        else:
            if np.isscalar(target_types) or isinstance(target_types, str):
                target_types = [target_types]
            target_indices = np.flatnonzero(np.isin(cell_types, list(target_types)))
    else:
        target_indices = np.flatnonzero(np.asarray(target_mask, dtype=bool))

    # --- build adjacency (sparse CSR), possibly per slide ---
    rows, cols, data = [], [], []

    if (slide_ids is not None) and same_slide:
        slide_ids = np.asarray(slide_ids)
        if slide_ids.shape[0] != n:
            raise ValueError("slide_ids must have length n.")
        unique_slides = pd.unique(slide_ids)

        for s in unique_slides:
            idx = np.flatnonzero(slide_ids == s)
            if idx.size == 0:
                continue
            Xs = coords[idx]

            if mode == "radius":
                # choose slide-specific radius if provided
                rad_s = (radius_map.get(s) if (radius_map is not None and s in radius_map) else radius)
                if not (isinstance(rad_s, (int, float)) and rad_s > 0):
                    raise ValueError(f"Provide a positive radius for slide {s} (or a global `radius`).")
                A_s = radius_neighbors_graph(
                    Xs, radius=rad_s, mode="connectivity",
                    include_self=include_self, metric=metric
                )
            elif mode == "knn":
                k_s = int(k_map.get(s) if (k_map is not None and s in k_map) else (k if k is not None else 0))
                if not (isinstance(k_s, int) and k_s > 0):
                    raise ValueError(f"Provide a positive integer k for slide {s} (or a global `k`).")
                # sklearn requires n_neighbors <= n_samples - (0 or 1)
                n_s = Xs.shape[0]
                max_k = n_s if include_self else max(n_s - 1, 1)
                k_eff = min(k_s, max_k)
                A_s = kneighbors_graph(
                    Xs, n_neighbors=k_eff, mode="connectivity",
                    include_self=include_self, metric=metric
                )
            else:
                raise ValueError("mode must be 'radius' or 'knn'.")

            coo = A_s.tocoo()
            # map local indices to global
            rows.append(idx[coo.row])
            cols.append(idx[coo.col])
            data.append(coo.data)

        if len(rows) == 0:
            # no edges at all (e.g., radius too small everywhere)
            A = csr_matrix((n, n), dtype=np.int8)
        else:
            rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
            A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int8)

    else:
        # global (ignoring slides)
        if mode == "radius":
            if not (isinstance(radius, (int, float)) and radius > 0):
                raise ValueError("Provide a positive `radius` when mode='radius'.")
            A = radius_neighbors_graph(
                coords, radius=radius, mode="connectivity",
                include_self=include_self, metric=metric
            )
        elif mode == "knn":
            if not (isinstance(k, int) and k > 0):
                raise ValueError("Provide a positive integer `k` when mode='knn'.")
            # ensure k is valid given n
            max_k = n if include_self else max(n - 1, 1)
            k_eff = min(k, max_k)
            A = kneighbors_graph(
                coords, n_neighbors=k_eff, mode="connectivity",
                include_self=include_self, metric=metric
            )
        else:
            raise ValueError("mode must be 'radius' or 'knn'.")

    # --- counts per type via sparse matmul ---
    H = csr_matrix(
        (np.ones(n, dtype=np.int32), (np.arange(n), type_codes)),
        shape=(n, m)
    )
    C = (A @ H)  # (n x m) sparse counts

    # slice to targets and build DataFrame
    data_arr = C[target_indices].toarray().astype(np.int32)
    df = pd.DataFrame(data_arr, columns=types.tolist())

    # set friendly index
    if ids is not None:
        ids = np.asarray(ids)
        df.index = ids[target_indices]
    else:
        df.index = target_indices

    return df


def cell_neighbor_treatment(
    coords: pd.DataFrame,                     # index = cell IDs; columns ['x','y'] (and/or ['z'])
    target_ids: Optional[Iterable[str]] = None,
    treatment: Optional[pd.Series] = None,          # index = cell IDs; True if a cell is “treatment” type
    *,
    radius: Optional[float] = None,           # choose exactly ONE of {radius, k}
    k: Optional[int] = None,
    metric: str = "euclidean",
    same_slide: bool = False,
    slide_ids: Optional[pd.Series] = None,    # index=cell IDs; required if same_slide=True
    exclude_self: bool = True,
    weight: str = "binary",                   # 'binary' or 'gaussian'
    bandwidth: Optional[float] = None,        # used when weight='gaussian' (default inferred)
    include_treatment_in_matrix: bool = False,# if False: treatment neighbors are removed from matrix (recommended)
    normalize_columns: bool = False,          # renormalize each target column to sum 1 (if >0)
    dtype: str = "float32",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build in a single pass:
      - neighbor_bin: (cells × targets) sparse DataFrame of neighbor weights
      - T:            (targets,) Series; T[target]=1 if any neighbor is a treatment cell

    Logic:
      • Neighbors are selected by `radius` OR `k` (choose one).
      • T is set to 1 if any neighbor ∈ treatment mask (checked BEFORE optional exclusion).
      • If include_treatment_in_matrix=False, those treatment neighbors are excluded from neighbor_bin.
      • If weight='gaussian', weights = exp(-d^2 / (2*bandwidth^2)).
      • If normalize_columns=True, each non-empty target column is scaled to sum to 1.

    Returns
    -------
    neighbor_bin : pd.DataFrame (sparse), index=cells, columns=target_ids
    T            : pd.Series of {0,1}, index=target_ids
    """
    # --- validations & alignment ---
    if (radius is None) == (k is None):
        raise ValueError("Specify exactly one of `radius` or `k`.")
    if not isinstance(coords, pd.DataFrame) or coords.shape[1] < 2:
        raise ValueError("`coords` must be a DataFrame with ≥2 spatial columns and cell IDs as index.")
    if same_slide and slide_ids is None:
        raise ValueError("`slide_ids` is required when same_slide=True.")
    if treatment is not None and not treatment.index.equals(coords.index):
        treatment = treatment.reindex(coords.index).fillna(False)

    all_ids = coords.index.to_list()
    if target_ids is None:
        target_ids = all_ids
    else:
        target_ids = [t for t in target_ids if t in coords.index]  # keep order, drop missing

    X = coords.values.astype(np.float32)
    t_idx = np.array([coords.index.get_loc(t) for t in target_ids], dtype=int)
    T_xy = X[t_idx]

    # --- neighbor search (kNN or radius) ---
    nn = NearestNeighbors(
        n_neighbors=(k + 1 if k is not None else None),  # +1 so we can drop self
        radius=radius if radius is not None else None,
        metric=metric
    ).fit(X)

    if radius is not None:
        dlist, ilist = nn.radius_neighbors(T_xy, radius=radius, return_distance=True)
    else:
        dlist, ilist = nn.kneighbors(T_xy, return_distance=True)
        # drop self neighbor if present
        if exclude_self:
            for i in range(len(ilist)):
                if ilist[i][0] == t_idx[i]:
                    ilist[i] = ilist[i][1:]
                    dlist[i] = dlist[i][1:]

    # default bandwidth for gaussian weights
    if weight == "gaussian" and bandwidth is None:
        all_d = np.concatenate([d[d > 0] for d in dlist if len(d) > 0])
        bandwidth = np.median(all_d) if all_d.size else 1.0

    # slide & treatment vectors for quick masking
    slide_arr = slide_ids.reindex(coords.index).to_numpy() if same_slide else None
    treat_arr = treatment.to_numpy()

    rows, cols, data = [], [], []
    T_vals = np.zeros(len(t_idx), dtype=np.int8)

    # --- single pass over targets ---
    for col_j, (nbr_idx, dists) in enumerate(zip(ilist, dlist)):
        if len(nbr_idx) == 0:
            continue

        # same-slide filter
        if same_slide:
            tgt_slide = slide_arr[t_idx[col_j]]
            keep = (slide_arr[nbr_idx] == tgt_slide)
            nbr_idx = nbr_idx[keep]
            dists   = dists[keep]
            if len(nbr_idx) == 0:
                continue

        # drop self just in case
        if exclude_self:
            mask = (nbr_idx != t_idx[col_j])
            nbr_idx = nbr_idx[mask]
            dists   = dists[mask]
            if len(nbr_idx) == 0:
                continue

        # treatment detection (for T)
        is_tr = treat_arr[nbr_idx]
        T_vals[col_j] = 1 if np.any(is_tr) else 0

        # optionally exclude treatment neighbors from matrix (so X is treatment-free)
        if not include_treatment_in_matrix:
            keep = ~is_tr
            nbr_idx = nbr_idx[keep]
            dists   = dists[keep]
            if len(nbr_idx) == 0:
                continue

        # weights
        if weight == "binary":
            vals = np.ones(len(nbr_idx), dtype=dtype)
        elif weight == "gaussian":
            vals = np.exp(-(dists.astype(np.float32)**2) / (2.0 * (bandwidth**2))).astype(dtype)
        else:
            raise ValueError("weight must be 'binary' or 'gaussian'.")

        rows.extend(nbr_idx.tolist())
        cols.extend([col_j] * len(nbr_idx))
        data.extend(vals.tolist())

    # build sparse matrix with full shape even if some columns are empty
    M = csr_matrix((np.asarray(data, dtype=dtype), (np.asarray(rows), np.asarray(cols))),
                   shape=(coords.shape[0], len(t_idx)))

    # renormalize columns if requested
    if normalize_columns:
        col_sums = np.asarray(M.sum(axis=0)).ravel()
        nz = col_sums > 0
        if nz.any():
            scale = np.zeros_like(col_sums, dtype=np.float32)
            scale[nz] = 1.0 / col_sums[nz]
            M = M @ diags(scale, format="csr")

    nb = pd.DataFrame.sparse.from_spmatrix(
        M, index=coords.index, columns=pd.Index(target_ids, name="target")
    )
    T = pd.Series(T_vals.astype(int), index=pd.Index(target_ids, name="target"), name="T")
    return nb, T