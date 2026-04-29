# neighbor_embedding.py
import os, pickle, math
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import multiprocessing as mp

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import linprog

from . import BuildGraph as bg


# ------------- Extract supports/weights from neighbor_bin -------------
def _get_supports_and_weights(neighbor_bin: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    target_names = list(neighbor_bin.columns)
    supps, weights = [], []

    if pd.api.types.is_sparse(neighbor_bin.dtypes.iloc[0]):
        A = neighbor_bin.sparse.to_coo().tocsc().astype(np.float64)
        for j in range(A.shape[1]):
            start, end = A.indptr[j], A.indptr[j+1]
            idx = A.indices[start:end]
            vals = A.data[start:end]
            if idx.size == 0:
                supps.append(np.array([], dtype=int))
                weights.append(np.array([], dtype=float))
            else:
                w = vals.astype(np.float64)
                w = w / w.sum()
                supps.append(idx)
                weights.append(w)
    else:
        M = neighbor_bin.to_numpy(dtype=np.float64)
        for j in range(M.shape[1]):
            col = M[:, j]
            idx = np.where(col > 0)[0]
            if idx.size == 0:
                supps.append(np.array([], dtype=int))
                weights.append(np.array([], dtype=float))
            else:
                w = col[idx].astype(np.float64)
                w = w / w.sum()
                supps.append(idx)
                weights.append(w)

    return supps, weights, target_names


# ---------------------- Exact EMD cost via LP ----------------------
def _emd_cost_linprog(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    m, n = C.shape
    c = C.ravel()
    A_eq = np.zeros((m + n, m * n), dtype=np.float64)
    for i in range(m):
        A_eq[i, i*n:(i+1)*n] = 1.0
    for j in range(n):
        A_eq[m + j, j::n] = 1.0
    b_eq = np.concatenate([a, b]).astype(np.float64)
    bounds = [(0.0, None)] * (m * n)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")
    return float(res.fun)


# ================= Multiprocessing worker state =================
_G = None
_supps = None
_weights = None
_target_names = None
_exact_mode = False

def _init_worker(G, supps, weights, target_names, exact):
    # Set globals once per worker
    global _G, _supps, _weights, _target_names, _exact_mode
    _G = G
    _supps = supps
    _weights = weights
    _target_names = target_names
    _exact_mode = exact


def _process_anchor_block(i_start: int, i_end: int) -> List[Tuple[int, int, float]]:
    """
    Process anchors i in [i_start, i_end) inclusive-exclusive.
    For each anchor i:
      - run one Dijkstra from supp(i) to all nodes (multi-source)
      - compute distances to all j > i
    Returns list of (i, j, dis)
    """
    out: List[Tuple[int, int, float]] = []
    n_targets = len(_target_names)

    for i in range(i_start, i_end):
        idx_i = _supps[i]
        if idx_i.size == 0:
            # Record NaNs for all pairs (i, j>i)
            out.extend((i, j, np.nan) for j in range(i+1, n_targets))
            continue

        # One Dijkstra per anchor i: shape (m_i, n_cells)
        D_i = dijkstra(_G, directed=False, indices=idx_i)  # float64
        a = _weights[i]

        if not _exact_mode:
            # Precompute column-wise minima once: min over supp(i) to each node v
            colmin_all = D_i.min(axis=0)  # (n_cells,)
            for j in range(i+1, n_targets):
                idx_j = _supps[j]
                if idx_j.size == 0:
                    out.append((i, j, np.nan)); continue
                b = _weights[j]

                # forward (i->j): for each s in supp(i), min over t in supp(j) D_i[s, t]
                rowmin_to_j = D_i[:, idx_j].min(axis=1)    # (m_i,)
                d_fwd = float(np.dot(a, rowmin_to_j))

                # backward (j->i): for each t in supp(j), min over s in supp(i) D_i[s, t]
                colmin_from_j = colmin_all[idx_j]          # (m_j,)
                d_bwd = float(np.dot(b, colmin_from_j))

                out.append((i, j, 0.5 * (d_fwd + d_bwd)))
        else:
            # Exact EMD: C = D_i[:, idx_j]
            for j in range(i+1, n_targets):
                idx_j = _supps[j]
                if idx_j.size == 0:
                    out.append((i, j, np.nan)); continue
                b = _weights[j]
                C = D_i[:, idx_j]  # (m_i, m_j)
                d = _emd_cost_linprog(a, b, C)
                out.append((i, j, d))

        # free D_i for this anchor before next i
        del D_i

    return out


# ======================== Public API ========================
def neighbor_distance(
    embedding: pd.DataFrame,
    neighbor_bin: pd.DataFrame,
    outpath: str,
    graph_k: int = 30,
    overwrite: bool = False,
    exact: bool = False,
    cores: int = 1,
    start_method: Optional[str] = None,  # 'fork', 'spawn', or None=auto
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Multi-core, memory-savvy neighbor distance:
      * upper-triangle only
      * one Dijkstra per anchor target
      * frees per-anchor distance matrix immediately

    exact=False : NN-surrogate (symmetric avg of bidirectional nearest-distance)
    exact=True  : EMD via linear program

    Returns: DataFrame columns ["target1", "target2", "dis"].
    """
    Path(outpath).mkdir(parents=True, exist_ok=True)
    graph_file = os.path.join(outpath, "cell_graph.pkl")
    dis_file   = os.path.join(outpath, "neighbor_graph_dis.pkl")

    if not isinstance(embedding, pd.DataFrame):
        raise ValueError("embedding must be a pandas DataFrame (cells as index).")
    if not isinstance(neighbor_bin, pd.DataFrame):
        raise ValueError("neighbor_bin must be a pandas DataFrame (cells as index, targets as columns).")

    # Align rows
    cells = embedding.index.intersection(neighbor_bin.index)
    if len(cells) < len(neighbor_bin.index) and verbose:
        print(f"[warn] Dropping {len(neighbor_bin.index) - len(cells)} rows not found in embedding.")
    E = embedding.loc[cells].to_numpy(dtype=np.float32)
    NB = neighbor_bin.loc[cells]

    # Load/build graph
    if os.path.exists(graph_file) and not overwrite:
        if verbose: print("[info] Loading cached SNN graph.")
        with open(graph_file, "rb") as f: G = pickle.load(f)
    else:
        if verbose: print(f"[info] Building SNN graph (k={graph_k}) ...")
        G = bg.build_connected_hybrid_graph(E, k_snn=graph_k)
        with open(graph_file, "wb") as f: pickle.dump(G, f)

    # Cached distances?
    if os.path.exists(dis_file) and not overwrite:
        if verbose: print("[info] Loading cached neighbor distances.")
        with open(dis_file, "rb") as f: return pickle.load(f)

    # Supports & weights
    supps, weights, target_names = _get_supports_and_weights(NB)
    n_targets = len(target_names)
    if verbose:
        print(f"[info] {E.shape[0]} cells, {E.shape[1]}-D embed; {n_targets} target cells.")
        mode = "Exact EMD" if exact else "NN-surrogate"
        print(f"[info] {mode}; upper-triangle pairs: {n_targets*(n_targets-1)//2:,}")

    # Multiprocessing setup
    if start_method is None:
        # HPC/Linux default is usually 'fork' (best).
        # On macOS/Python3.8+ and some clusters it's 'spawn'—still OK, just more pickling.
        start_method = mp.get_start_method(allow_none=True) or ("fork" if hasattr(mp, "get_context") else "spawn")
    ctx = mp.get_context(start_method)

    # Split anchors into blocks to distribute across workers
    cores = max(1, int(cores))
    n_blocks = min(cores * 4, n_targets)  # a few blocks per core for load balancing
    block_size = math.ceil(n_targets / n_blocks)
    blocks = [(i, min(i + block_size, n_targets)) for i in range(0, n_targets, block_size)]

    if cores == 1:
        _init_worker(G, supps, weights, target_names, exact)
        results_nested = [_process_anchor_block(b0, b1) for (b0, b1) in blocks]
    else:
        with ctx.Pool(
            processes=cores,
            initializer=_init_worker,
            initargs=(G, supps, weights, target_names, exact)
        ) as pool:
            results_nested = pool.starmap(_process_anchor_block, blocks)

    # Flatten & assemble
    rows = []
    for chunk in results_nested:
        rows.extend(chunk)

    out = pd.DataFrame(rows, columns=["i", "j", "dis"])
    # map to names
    out["target1"] = [target_names[i] for i in out["i"].to_numpy()]
    out["target2"] = [target_names[j] for j in out["j"].to_numpy()]
    out = out[["target1", "target2", "dis"]].copy()

    with open(dis_file, "wb") as f: pickle.dump(out, f)
    if verbose: print("[info] Done.")
    return out
