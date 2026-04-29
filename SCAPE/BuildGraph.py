import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

def build_snn_graph(embedding: np.ndarray, k: int = 10) -> csr_matrix:
    n = embedding.shape[0]
    k = min(k, max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(embedding)
    dists, knn = nn.kneighbors(embedding, return_distance=True)
    if knn.shape[1] and np.any(knn[:, 0] == np.arange(n)):
        knn, dists = knn[:, 1:], dists[:, 1:]
    # SNN weight via shared-NN count; convert to a "cost" ~ exp(-shared)
    knn_sets = [set(row) for row in knn]
    rows, cols, data = [], [], []
    
    for i in range(n):
        Ni = knn_sets[i]
        for j_idx, (j, d_ij) in enumerate(zip(knn[i], dists[i])):
            if j <= i: continue
            shared = len(Ni.intersection(knn_sets[j]))
            sn = (shared + 0.5) / (k + 0.5)
            # allow shared=0 edges to improve connectivity
            # cost = float(np.exp(-shared))
            cost = float(d_ij / (sn))
            rows += [i, j]; cols += [j, i]; data += [cost, cost]
    return csr_matrix((np.array(data, np.float32), (np.array(rows), np.array(cols))), shape=(n, n))

def build_euclidean_knn_graph(embedding: np.ndarray, k: int = 2, scale: Optional[float] = None) -> csr_matrix:
    n = embedding.shape[0]
    k = min(k, max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(embedding)
    dists, knn = nn.kneighbors(embedding, return_distance=True)
    knn, dists = knn[:, 1:], dists[:, 1:]  # drop self
    if scale is None:
        # scale so typical kNN edge costs are roughly O(1), comparable to SNN exp costs
        scale = np.median(dists)
        if scale <= 0: scale = 1.0
    rows, cols, data = [], [], []
    for i in range(n):
        for j, d in zip(knn[i], dists[i]):
            rows += [i, j]; cols += [j, i]; data += [float(d/scale), float(d/scale)]
    return csr_matrix((np.array(data, np.float32), (np.array(rows), np.array(cols))), shape=(n, n))

def union_graph_min(G1: csr_matrix, G2: csr_matrix) -> csr_matrix:
    # union with elementwise min on overlapping edges; keeps the nonzero where only one has it
    A = G1.tocoo(); B = G2.tocoo()
    rows = np.concatenate([A.row, B.row]); cols = np.concatenate([A.col, B.col]); data = np.concatenate([A.data, B.data])
    # consolidate duplicates by min
    import pandas as pd
    df = pd.DataFrame({'r': rows, 'c': cols, 'w': data})
    df = df.groupby(['r', 'c'], as_index=False)['w'].min()
    return csr_matrix((df['w'].to_numpy(), (df['r'].to_numpy(), df['c'].to_numpy())), shape=G1.shape)

def connect_components_with_mst(embedding: np.ndarray, 
                                G: csr_matrix,
                                k_snn: int = 30) -> csr_matrix:
    # If G is disconnected, add one Euclidean edge between the closest pair of points
    # for each edge in the MST over components (built on centroid distances).
    n_comp, labels = connected_components(G, directed=False)
    if n_comp <= 1: return G
    print(f"[warn] The SNN graph is not fully connected with {n_comp} compartments. Compartments will be connected by MST.")
    comps = [np.where(labels == c)[0] for c in range(n_comp)]
    centroids = np.stack([embedding[idx].mean(axis=0) for idx in comps], axis=0)
    # dense component distance matrix and its MST
    from sklearn.metrics import pairwise_distances
    Dc = pairwise_distances(centroids, metric='euclidean')
    np.fill_diagonal(Dc, 0.0)
    C = csr_matrix(Dc)
    mst = minimum_spanning_tree(C).tocoo()
    # For each MST edge (u,v), add the closest across-component pair
    rows, cols, data = [], [], []

    denom = 0.5/(k_snn+0.5)
    for u, v in zip(mst.row, mst.col):
        I = comps[int(u)]; J = comps[int(v)]
        tree = KDTree(embedding[J])
        d, j_idx = tree.query(embedding[I], k=1, return_distance=True)
        m = int(np.argmin(d))
        i_star = I[m]; j_star = J[int(j_idx[m, 0])]
        w = float(d[m])/denom
        rows += [i_star, j_star]; cols += [j_star, i_star]; data += [w, w]
    add = csr_matrix((np.array(data, np.float32), (np.array(rows), np.array(cols))), shape=G.shape)
    return union_graph_min(G, add)

def _as_array(embedding) -> np.ndarray:
    # accepts DataFrame, ndarray, or AnnData
    if isinstance(embedding, np.ndarray):
        return embedding.astype(np.float32, copy=False)
    if isinstance(embedding, pd.DataFrame):
        return embedding.to_numpy(dtype=np.float32)
    # optional: anndata support
    if hasattr(embedding, "X"):
        return np.asarray(embedding.X, dtype=np.float32)
    raise TypeError("`embedding` must be a numpy array, pandas DataFrame, or AnnData.")

def build_connected_hybrid_graph(embedding,
                                 k_snn: int = 30, k_backbone: int = 2) -> csr_matrix:
    X = _as_array(embedding)
    G_snn = build_snn_graph(X, k=k_snn)
    # G_knn = build_euclidean_knn_graph(X, k=k_backbone)
    # G = union_graph_min(G_snn, G_knn)
    # final safety: connect components via centroid MST bridges
    G = connect_components_with_mst(X, G_snn)
    return G
