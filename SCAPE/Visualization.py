import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import umap  # pip install umap-learn
import igraph as ig
import matplotlib.pyplot as plt

# Optional: for true Leiden with resolution parameter
try:
    import leidenalg
    _HAS_LEIDENALG = True
except ImportError:
    _HAS_LEIDENALG = False


# --------------------- theme_pre equivalent (matplotlib) --------------------- #
def theme_pre():
    """
    Rough analogue of your ggplot2 theme_pre for matplotlib.
    Call at the start of your plotting code.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.titlesize": 22,
        "axes.titleweight": "bold",
        "axes.titlelocation": "center",
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.title_fontsize": 16,
        "legend.fontsize": 16,
    })


def scatter(
    coords,
    group=None,
    treatment=None,
    group_labels=None,
    group_title="group",
    treatment_label="Treated",
    point_size=4,
    highlight_size=14,
    palette="publication",
    cmap="viridis",
    continuous=False,
    width=180,
    height=180,
    ax=None,           # pass an existing Axes to embed in a subplot grid
    rasterized=None,   # None = auto (rasterize if n_points > raster_thresh)
    raster_thresh=50000,
    show_legend=True,  # set False when combining panels (legend drawn by combine_scatters)
    show_ticks=True,   # set False to hide axis ticks and tick labels
    show_labels=True,  # set False to hide axis labels ("Dim 1" / "Dim 2")
    title=None,        # optional figure title
    xlab_label = "Dim1",
    ylab_label = "Dim2"
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    # ── publication style (Nature column widths) ──────────────────────────────
    mm_to_in = 1 / 25.4
    # widths = {"single": 89 * mm_to_in, "double": 183 * mm_to_in}

    mpl.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":            12,
        "axes.labelsize":       14,
        "axes.titlesize":       22,
        "axes.titleweight":     "bold",
        "xtick.labelsize":      18,
        "ytick.labelsize":      18,
        "legend.fontsize":      12,
        "legend.title_fontsize": 16,
        "axes.linewidth":       0.6,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
        "xtick.major.size":     2.5,
        "ytick.major.size":     2.5,
        "pdf.fonttype":         42,   # editable text in Illustrator
        "ps.fonttype":          42,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=(width * mm_to_in, height * mm_to_in))
        own_fig = True
    else:
        fig = ax.figure
        own_fig = False

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=2.5, width=0.6)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    # ── group setup ───────────────────────────────────────────────────────────
    if group is None:
        group = np.zeros(coords.shape[0])
    group = np.asarray(group)

    # auto-detect continuous: must be float AND have many unique values
    if not continuous:
        continuous = (
            np.issubdtype(group.dtype, np.floating)
            and len(np.unique(group)) > 10
        )

    # auto-detect rasterization based on point count
    if rasterized is None:
        rasterized = coords.shape[0] > raster_thresh

    # ── Paul Tol colorblind-safe bright palette ───────────────────────────────
    PUBLICATION_COLORS = [
        "#4477AA",  # blue
        "#EE6677",  # red
        "#228833",  # green
        "#CCBB44",  # yellow
        "#66CCEE",  # cyan
        "#AA3377",  # purple
        "#BBBBBB",  # grey
        "#EE8866",  # orange
        "#44BB99",  # teal
        "#FFAABB",  # pink
    ]

    legend_handles = []

    # ── continuous mode ───────────────────────────────────────────────────────
    if continuous:
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=group, cmap=cmap,
            s=point_size, alpha=0.8,
            linewidths=0, rasterized=rasterized, zorder=2,
        )
        if show_legend:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                                shrink=0.8, aspect=20)
            cbar.set_label(group_title)
            cbar.ax.tick_params(width=0.5, length=2)
            cbar.outline.set_linewidth(0.5)

    # ── discrete mode ─────────────────────────────────────────────────────────
    else:
        colors = PUBLICATION_COLORS if palette == "publication" \
                 else list(plt.cm.tab10.colors)
        unique_groups = np.unique(group)

        for i, g in enumerate(unique_groups):
            mask  = group == g
            label = group_labels[i] if group_labels is not None else str(g)
            color = colors[i % len(colors)]

            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=point_size, color=color, alpha=0.8,
                linewidths=0, rasterized=rasterized, zorder=2,
            )
            legend_handles.append(
                Line2D([0], [0], marker="o", color="none",
                       markerfacecolor=color, markeredgewidth=0,
                       markersize=4, label=label)
            )

    # ── treatment overlay ─────────────────────────────────────────────────────
    if treatment is not None:
        treat_mask = np.asarray(treatment).astype(bool)
        ax.scatter(
            coords[treat_mask, 0], coords[treat_mask, 1],
            s=highlight_size, facecolors="none",
            edgecolors="#222222", linewidths=0.5,
            rasterized=rasterized, zorder=3,
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="none", markeredgecolor="#222222",
                   markeredgewidth=0.8, markersize=5,
                   label=treatment_label)
        )

    # ── store handles on axes for retrieval by combine_scatters ──────────────
    ax._scape_legend_handles = legend_handles
    ax._scape_legend_title   = group_title

    # ── legend (discrete + optional treatment) ────────────────────────────────
    if legend_handles and show_legend:
        leg = ax.legend(
            handles=legend_handles,
            title=group_title,
            frameon=False,
            handletextpad=0.3,
            borderpad=0.0,
            labelspacing=0.3,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
        leg.get_title().set_fontweight("bold")
        if own_fig:
            fig.tight_layout()
            fig.subplots_adjust(right=0.82)

    if show_labels:
        ax.set_xlabel(xlab_label)
        ax.set_ylabel(ylab_label)
    if title is not None:
        ax.set_title(title)

    return fig, ax


def combine_scatters(
    panels,
    ncols=None,
    titles=None,
    sharex=False,
    sharey=False,
    panel_size=(3.0, 3.0),
    wspace=0.35,
    hspace=0.4,
    legend_pos="right",   # "right", "bottom", or None
    legend_ncol=1,        # columns in legend (useful for "bottom" placement)
):
    """
    Lay out multiple scatter panels in a grid with a single merged legend.

    Parameters
    ----------
    panels : list of dict
        Each dict is kwargs forwarded to scatter() (do not include 'ax' or
        'show_legend' — those are handled internally).
    ncols : int, optional
        Number of columns. Defaults to len(panels) (single row).
    titles : list of str, optional
        Per-panel titles.
    sharex, sharey : bool
        Share axis limits across panels.
    panel_size : (float, float)
        (width, height) in inches per panel.
    wspace, hspace : float
        Horizontal / vertical spacing between panels (fraction of panel size).
    legend_pos : "right" | "bottom" | None
        Where to place the merged legend.
    legend_ncol : int
        Number of columns in the legend (handy when legend_pos="bottom").

    Returns
    -------
    fig : matplotlib.Figure
    axes : list of matplotlib.Axes
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    n = len(panels)
    if ncols is None:
        ncols = n
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        sharex=sharex, sharey=sharey,
        squeeze=False,
    )
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    axes_flat = axes.flatten()

    # hide unused panels
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # draw each panel
    for i, kwargs in enumerate(panels):
        scatter(**kwargs, ax=axes_flat[i], show_legend=False)
        if titles is not None and i < len(titles):
            axes_flat[i].set_title(titles[i])

    # ── collect and deduplicate legend handles across all panels ───────────────
    seen_labels  = {}   # label → handle (first occurrence wins)
    legend_title = ""

    for ax_i in axes_flat[:n]:
        handles = getattr(ax_i, "_scape_legend_handles", [])
        title_i = getattr(ax_i, "_scape_legend_title", "")
        if title_i:
            legend_title = title_i          # use last non-empty title
        for h in handles:
            if h.get_label() not in seen_labels:
                seen_labels[h.get_label()] = h

    merged_handles = list(seen_labels.values())

    if merged_handles and legend_pos is not None:
        if legend_pos == "right":
            leg = fig.legend(
                handles=merged_handles,
                title=legend_title,
                frameon=False,
                handletextpad=0.3,
                borderpad=0.0,
                labelspacing=0.3,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                ncol=legend_ncol,
            )
        elif legend_pos == "bottom":
            leg = fig.legend(
                handles=merged_handles,
                title=legend_title,
                frameon=False,
                handletextpad=0.3,
                borderpad=0.0,
                labelspacing=0.3,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=legend_ncol if legend_ncol > 1 else len(merged_handles),
            )
        leg.get_title().set_fontweight("bold")

    return fig, list(axes_flat[:n])
    
    
# -------------------------- umap_from_exprs ---------------------------------- #
def umap_from_exprs(exprs, n_pcs=30, n_neighbors=30, scale_genes=True, seed=1024):
    """
    Python version of umap_from_exprs.
    exprs: (n_cells, n_genes) numpy array
    Returns: pandas.DataFrame with columns ['umap1', 'umap2']
    """
    exprs = np.asarray(exprs, dtype=float)
    n_cells, n_genes = exprs.shape

    if scale_genes:
        # fast column-wise z-score
        cm = exprs.mean(axis=0)
        exprs = exprs - cm
        # unbiased variance denominator: max(n-1,1)
        denom = max(n_cells - 1, 1)
        cs = np.sqrt((exprs ** 2).sum(axis=0) / denom)
        cs[cs == 0] = 1.0
        exprs = exprs / cs

    r = min(n_pcs, n_cells - 1, n_genes)

    # Truncated SVD ~ IRLBA PCA on already centered/scaled matrix
    svd = TruncatedSVD(n_components=r, random_state=seed)
    PC = svd.fit_transform(exprs)  # (n_cells, r)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric="euclidean",
        random_state=seed,
        verbose=False,
    )
    U = reducer.fit_transform(PC)
    U_df = pd.DataFrame(U, columns=["umap1", "umap2"])
    return U_df


# ------------------------------- ridge --------------------------------------- #
def ridge(X, Y, lam=1e-2):
    """
    Python version of ridge().
    Solves (X, 1) -> Y with L2 penalty on coefficients (but not intercept).

    X: (n, p) numpy array
    Y: (n, d) numpy array
    lam: ridge penalty lambda

    Returns dict: {'A': A, 'b': b}
        A: (p, d) coefficient matrix
        b: (1, d) intercept row
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = X.shape

    X1 = np.hstack([X, np.ones((n, 1))])      # (n, p+1)
    XtX = X1.T @ X1                           # (p+1, p+1)
    XtY = X1.T @ Y                            # (p+1, d)

    pen = np.diag(np.concatenate([np.full(p, lam), [0.0]]))
    B = np.linalg.solve(XtX + pen, XtY)       # (p+1, d)

    A = B[:p, :]
    b = B[p:p+1, :]
    return {"A": A, "b": b}


# ---------------------------- delta_smooth ----------------------------------- #
def delta_smooth(f0_pc, Delta_pc, e_hat=None,
                 k=30, sigma_geom=None, sigma_e=None,
                 trim=(0.05, 0.95)):
    """
    Propensity-aware kernel smoothing on f0 space.
    Direct port of your R delta_smooth.
    
    f0_pc: (n, d)
    Delta_pc: (n, d)
    e_hat: (n,) or None
    Returns: (keep_idx, out)
      keep_idx: 1D array of indices that were kept (0-based)
      out: (n, d) array of smoothed deltas with NaN for non-kept rows
    """
    f0_pc = np.asarray(f0_pc, dtype=float)
    Delta_pc = np.asarray(Delta_pc, dtype=float)
    n, d = f0_pc.shape
    assert Delta_pc.shape == (n, d)

    if e_hat is not None:
        e_hat = np.asarray(e_hat, dtype=float)
        keep = (e_hat >= trim[0]) & (e_hat <= trim[1])
    else:
        keep = np.ones(n, dtype=bool)

    keep_idx = np.where(keep)[0]   # 0-based indices
    X = f0_pc[keep, :]             # (n_keep, d)
    D0 = Delta_pc[keep, :]         # (n_keep, d)

    # KNN on X
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
    dists, indices = nbrs.kneighbors(X)  # both (n_keep, k)

    if sigma_geom is None:
        sigma_geom = np.median(dists)

    if e_hat is not None and sigma_e is None:
        # IQR / 1.349 ~ std
        eh = e_hat[keep]
        q75, q25 = np.percentile(eh, [75, 25])
        iqr = q75 - q25
        sigma_e = iqr / 1.349 if iqr > 0 else 1.0

    out = np.full((n, d), np.nan, dtype=float)

    for row_local, (idx_row, dgeo) in enumerate(zip(indices, dists)):
        w = np.exp(-(dgeo ** 2) / (2 * sigma_geom ** 2))

        if e_hat is not None:
            eh_keep = e_hat[keep]
            de = np.abs(eh_keep[row_local] - eh_keep[idx_row])
            w *= np.exp(-(de ** 2) / (2 * sigma_e ** 2)) * \
                 (eh_keep[idx_row] * (1 - eh_keep[idx_row]))

        sw = w.sum()
        global_i = keep_idx[row_local]

        if sw > 0:
            # weighted average of D0[idx_row, :]
            out[global_i, :] = (w / sw) @ D0[idx_row, :]
        else:
            out[global_i, :] = 0.0

    return keep_idx, out


# ---------------------------- delta_match ------------------------------------ #
def delta_match(f0_pc, f1_pc, e_hat, k=30, trim=(0.05, 0.95)):
    """
    Python version of delta_match().
    f0_pc, f1_pc: (n, d) arrays in the same PC space.
    e_hat: (n,)
    Returns a pandas.DataFrame with columns:
      'untreat', 'treat', 'dist'
    using 0-based indices.
    """
    f0_pc = np.asarray(f0_pc, dtype=float)
    f1_pc = np.asarray(f1_pc, dtype=float)
    e_hat = np.asarray(e_hat, dtype=float)

    Delta_pc = f1_pc - f0_pc
    keep_idx, Delta_sm = delta_smooth(
        f0_pc=f0_pc, Delta_pc=Delta_pc, e_hat=e_hat,
        k=k, trim=trim
    )

    # Synthetic treated in PC space
    f1_synth_pc = f0_pc[keep_idx, :] + Delta_sm[keep_idx, :]

    # KNN: match synthetic treated to actual treated
    nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(f1_pc[keep_idx, :])
    dist, ind = nbrs.kneighbors(f1_synth_pc)
    match_j = ind[:, 0]        # local indices in keep_idx
    match_d = dist[:, 0]

    # map back to global indices
    treat_global = keep_idx[match_j]
    untreat_global = keep_idx

    df = pd.DataFrame({
        "untreat": untreat_global,
        "treat": treat_global,
        "dist": match_d,
    })
    return df


# -------------------------- leiden_embedding --------------------------------- #
def leiden_embedding(data, k=30, prune_snn=0.0, weight="jaccard", resolution=1.0):
    """
    Python port of your 'leiden_embedding' (actually Louvain on SNN).
    data: (n, d) array
    Returns: 1D numpy array of cluster labels (ints).
    
    Note: we implement SNN + Jaccard, then either Leiden (if leidenalg
    is available) or igraph's Louvain.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]

    # KNN to get neighbor indices
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(data)
    _, knn_idx = nbrs.kneighbors(data)  # (n, k), each row neighbors of i

    # Shared nearest neighbors:
    # For each node i, we know its neighbors knn_idx[i].
    # We'll build edges (i, j) where j in knn_idx[i], weight = jaccard.
    rows = []
    for i in range(n):
        neigh_i = knn_idx[i]
        set_i = set(neigh_i)
        for j in neigh_i:
            neigh_j = knn_idx[j]
            shared = len(set_i.intersection(neigh_j))
            jaccard = shared / (2 * k - shared) if (2 * k - shared) > 0 else 0.0
            if jaccard > prune_snn:
                rows.append((i, j, jaccard))

    if not rows:
        raise ValueError("No edges remain after SNN pruning; try lowering prune_snn.")

    edges_df = pd.DataFrame(rows, columns=["start", "end", "jaccard"])
    edges_df["weight"] = edges_df[weight]

    # Build igraph graph
    # igraph expects vertex IDs from 0..n-1; we already use that.
    g = ig.Graph(
        edges=list(zip(edges_df["start"].astype(int),
                       edges_df["end"].astype(int))),
        directed=False,
    )
    g.es["weight"] = edges_df["weight"].values

    if _HAS_LEIDENALG:
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=resolution,
        )
        membership = np.array(part.membership, dtype=int)
    else:
        # Fallback: Louvain (resolution ignored here)
        clusters = g.community_multilevel(weights=g.es["weight"])
        membership = np.array(clusters.membership, dtype=int)

    return membership


# ------------------------------ bin_vector ----------------------------------- #
def bin_vector(U, V, bins=40, min_n=10):
    """
    Python version of bin_vector().
    U: (n, 2) array of base positions (x, y)
    V: (n, 2) array of vectors (dx, dy)
    
    Returns a pandas.DataFrame with columns:
      x, y, xend, yend, n
    after binning and aggregation.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    assert U.shape == V.shape
    assert U.shape[1] == 2

    df = pd.DataFrame({
        "x": U[:, 0],
        "y": U[:, 1],
        "xend": V[:, 0],
        "yend": V[:, 1],
    })

    df["bx"] = pd.cut(df["x"], bins=bins)
    df["by"] = pd.cut(df["y"], bins=bins)

    grouped = (
        df.groupby(["bx", "by"], observed=True)
          .agg({
              "x": "median",
              "y": "median",
              "xend": "median",
              "yend": "median",
          })
          .reset_index()
    )
    grouped["n"] = df.groupby(["bx", "by"], observed=True).size().values

    # Convert xend/yend from vector to endpoint: x + dx, y + dy
    grouped["xend"] = grouped["x"] + grouped["xend"]
    grouped["yend"] = grouped["y"] + grouped["yend"]

    grouped = grouped[grouped["n"] >= min_n].reset_index(drop=True)

    # Keep only useful columns
    return grouped[["x", "y", "xend", "yend", "n"]]


# -------------------------- aggre_vector_field -------------------------------- #
def aggre_vector_field(U0, U1, keep=None, bins=40, min_n=10, k=3):
    """
    Python version of aggre_vector_field().

    U0, U1: (n, 2) arrays of coordinates (e.g., two embeddings).
    keep: optional boolean mask of length n; if provided, restricts to those rows.
          (In your R code, keep is passed but not used, so we ignore it as well.)
    bins, min_n: passed to bin_vector
    k: number of neighbors for mutual nearest neighbors.

    Returns:
      pandas.DataFrame with columns:
        U0, U1, dis_x (ignored; distances not kept here), U0_x, U0_y, U1_x, U1_y
      or None if no mutual pairs.
    """
    U0 = np.asarray(U0, dtype=float)
    U1 = np.asarray(U1, dtype=float)

    if U0.shape != U1.shape:
        raise ValueError("The rows between U0 and U1 should match!")

    # R code ignores 'keep' internally; we do as well.

    V = U1 - U0             # vector from U0 to U1

    U0_binv = bin_vector(U0, V, bins=bins, min_n=min_n)
    U1_binv = bin_vector(U1, -V, bins=bins, min_n=min_n)

    if U0_binv.empty or U1_binv.empty:
        import warnings
        warnings.warn("One of the binned vector fields is empty.")
        return None

    # KNN in binned space
    nbrs_01 = NearestNeighbors(n_neighbors=k).fit(U1_binv[["x", "y"]].to_numpy())
    dist_01, ind_01 = nbrs_01.kneighbors(U0_binv[["xend", "yend"]].to_numpy())

    nbrs_10 = NearestNeighbors(n_neighbors=k).fit(U0_binv[["x", "y"]].to_numpy())
    dist_10, ind_10 = nbrs_10.kneighbors(U1_binv[["xend", "yend"]].to_numpy())

    # Long format
    n0 = U0_binv.shape[0]
    n1 = U1_binv.shape[0]

    U0_U1_knn_long = pd.DataFrame({
        "U0": np.repeat(np.arange(n0), k),
        "U1": ind_01.reshape(-1),
        "dis": dist_01.reshape(-1),
    })

    U1_U0_knn_long = pd.DataFrame({
        "U0": ind_10.reshape(-1),
        "U1": np.repeat(np.arange(n1), k),
        "dis": dist_10.reshape(-1),
    })

    # Mutual nearest neighbors
    U0_U1_mnn = U0_U1_knn_long.merge(U1_U0_knn_long, on=["U0", "U1"], how="inner")

    if U0_U1_mnn.shape[0] > 0:
        # Attach coordinates
        U0_coords = U0_binv[["x", "y"]].rename(columns={"x": "U0_x", "y": "U0_y"})
        U1_coords = U1_binv[["x", "y"]].rename(columns={"x": "U1_x", "y": "U1_y"})

        out = U0_U1_mnn.merge(
            U0_coords.reset_index().rename(columns={"index": "U0"}),
            on="U0",
            how="left",
        ).merge(
            U1_coords.reset_index().rename(columns={"index": "U1"}),
            on="U1",
            how="left",
        )

        return out
    else:
        import warnings
        warnings.warn(
            "No mutual pairs could be found between U0 and U1 bins! "
            "Please check the correspondence of rows between U0 and U1 or increase k."
        )
        return None

def treatment_umap(y_z, m1, m0, y1, y0, treatment,
                   n_pcs=30, n_neighbors=30, seed=1024):
    """
    Python translation of the R function `treatment_umap`.

    Parameters
    ----------
    y_z : array-like, shape (n_samples, n_features)
        Observed expression (or embedding) under treatment assignment.
    m1, m0 : array-like, shape (n_samples, n_features)
        Nuisance models (e.g. predicted outcomes) under treated / untreated.
    y1, y0 : array-like, shape (n_samples, n_features)
        Potential outcome representations under treated / untreated regimes.
    treatment : array-like, shape (n_samples,)
        0/1 treatment indicator.
    n_pcs : int
        Maximum number of PCs to keep (default 30).
    n_neighbors : int
        UMAP neighbors (default 30).
    seed : int
        Random seed for UMAP and randomized SVD.

    Returns
    -------
    dict with keys:
        - 'exprs'     : DataFrame of UMAP of y_star_pc (observed / debiased)
        - 'untreated' : DataFrame of UMAP of counterfactual untreated
        - 'treated'   : DataFrame of UMAP of counterfactual treated
    """
    # Convert to numpy arrays
    y_z = np.asarray(y_z, dtype=float)
    m1 = np.asarray(m1, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    treatment = np.asarray(treatment)

    # Indices for treated / untreated
    idx0 = np.where(treatment == 0)[0]
    idx1 = np.where(treatment == 1)[0]

    # gamma = m1 - m0
    gamma = m1 - y1

    # y_star = y_z - gamma
    y_star = y_z - gamma

    n, p = y_star.shape
    r = min(n_pcs, n - 1, p)

    # PCA via TruncatedSVD on centered data
    center = y_star.mean(axis=0)
    y_star_centered = y_star - center

    svd = TruncatedSVD(n_components=r, random_state=seed)
    y_star_pc = svd.fit_transform(y_star_centered)  # (n, r)
    rotation = svd.components_.T                    # (p, r)

    # Function to project new data into this PC space (toPC in R)
    def toPC(M):
        M = np.asarray(M, dtype=float)
        return (M - center) @ rotation

    y0pc = toPC(y0)   # (n, r)
    y1pc = toPC(y1)   # (n, r)

    # Ridge calibration on each arm
    # (uses the earlier Python ridge() implementation)
    cal0 = ridge(y0pc[idx0, :], y_star_pc[idx0, :], lam=1e-2)
    cal1 = ridge(y1pc[idx1, :], y_star_pc[idx1, :], lam=1e-2)

    # Transform counterfactuals into factual PC domains
    # R: y1pc %*% cal1$A + matrix(rep(cal1$b, each=nrow(y1pc)), ...)
    y1pc_to_treated = y1pc @ cal1["A"] + cal1["b"]  # broadcasting (n, r)
    y0pc_to_untreated = y0pc @ cal0["A"] + cal0["b"]

    # UMAP on y_star_pc, then transform the calibrated counterfactual PCs
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric="euclidean",
        random_state=seed,
        verbose=False,
    )
    U_Y = reducer.fit_transform(y_star_pc)
    U_f1 = reducer.transform(y1pc_to_treated)
    U_f0 = reducer.transform(y0pc_to_untreated)

    # Wrap in DataFrames
    U_Y_df = pd.DataFrame(U_Y, columns=["umap1", "umap2"])
    U_f0_df = pd.DataFrame(U_f0, columns=["umap1", "umap2"])
    U_f1_df = pd.DataFrame(U_f1, columns=["umap1", "umap2"])

    return {
        "exprs": U_Y_df,
        "untreated": U_f0_df,
        "treated": U_f1_df,
    }


def connectivity_coord(coord, connectivity, dims=(0, 1)):
    """
    Python translation of the R function connectivity_coord.

    Parameters
    ----------
    coord : pd.DataFrame
        Coordinate table of size (n, d). We will use only columns given by `dims`.
    connectivity : array-like or scipy.sparse matrix, shape (n, n)
        Connectivity matrix between the n groups / clusters.
    dims : tuple of length 2
        Indices (0-based) or column names of `coord` to use as (x, y) coordinates.

    Returns
    -------
    pd.DataFrame
        Columns:
          - i, j : integer indices (0-based)
          - x    : connectivity value
          - i_x, i_y : coordinates of node i
          - j_x, j_y : coordinates of node j
    """
    # Ensure coord is a DataFrame
    if not isinstance(coord, pd.DataFrame):
        coord = pd.DataFrame(coord)

    n = coord.shape[0]

    # Convert connectivity to sparse matrix
    if sparse.issparse(connectivity):
        C = connectivity.copy()
    else:
        C = sparse.csr_matrix(np.asarray(connectivity))

    if C.shape[0] != C.shape[1] or C.shape[0] != n:
        raise ValueError("The number of points in coord and the connectivity doesn't match!")

    # Only 2D allowed (as in R code)
    if len(dims) != 2:
        raise ValueError("This function is for plotting and only 2 dimensions are allowed.")

    # Handle dims as indices or names
    if isinstance(dims[0], (int, np.integer)):
        cols = [coord.columns[d] for d in dims]
    else:
        cols = list(dims)

    coord2 = coord.loc[:, cols].copy()
    coord2.columns = ["x", "y"]
    coord2["id"] = np.arange(n)  # 0-based indexing

    # Set diagonal of connectivity to 0 (no self-edges)
    C = C.tolil()
    C.setdiag(0)
    C = C.tocsr()

    # Extract non-zero entries: rows i, cols j, data x
    C_coo = C.tocoo()
    df_conn = pd.DataFrame({
        "i": C_coo.row,   # 0-based
        "j": C_coo.col,   # 0-based
        "x": C_coo.data,
    })

    # Join coordinates for i
    coord_i = coord2.rename(columns={"x": "i_x", "y": "i_y"})
    df_conn = df_conn.merge(
        coord_i[["id", "i_x", "i_y"]].rename(columns={"id": "i"}),
        on="i",
        how="left"
    )

    # Join coordinates for j
    coord_j = coord2.rename(columns={"x": "j_x", "y": "j_y"})
    df_conn = df_conn.merge(
        coord_j[["id", "j_x", "j_y"]].rename(columns={"id": "j"}),
        on="j",
        how="left"
    )

    return df_conn

def dimplot(
    embedding,
    annot,
    color_by,
    alpha_by=None,
    connectivity=None,
    label=True,
    dims=(0, 1),               # can be col indices or names
    connectivity_thresh=0.1,
    label_size=10,
    label_type="text",         # "text" or "label"
    label_color="black",
    box_padding=0.25,          # kept for API symmetry; not used directly
    point_padding=1e-6,        # kept for API symmetry; not used directly
    raster_thresh=10000,
    ax=None,
    cmap="viridis",
    **scatter_kwargs,
):
    """
    Python translation of your R dimplot().

    Parameters
    ----------
    embedding : pd.DataFrame or array-like (n, d)
        Low-dimensional embedding (e.g., UMAP, PCA).
    annot : pd.DataFrame
        Annotation table with at least columns [color_by] and optionally [alpha_by],
        same index (rows) as embedding.
    color_by : str
        Column in annot used for color.
    alpha_by : str or None
        Column in annot used for per-point alpha (if not None).
    connectivity : pd.DataFrame or None
        Edge coordinate table with columns ['x', 'i_x', 'i_y', 'j_x', 'j_y'],
        typically output of a Python analogue of connectivity_coord().
    label : bool
        Whether to draw group labels at medians.
    dims : tuple
        Which dimensions of embedding to plot. Can be:
          - int indices (0-based)
          - column names of embedding (if embedding is a DataFrame)
    connectivity_thresh : float
        Minimum edge weight x to display.
    label_size : float
        Font size for labels.
    label_type : {"text", "label"}
        If "label", draw a box behind the text.
    label_color : str
        Text color.
    raster_thresh : int
        If number of points > raster_thresh, scatter is rasterized for speed.
    ax : matplotlib.axes.Axes or None
        If provided, plot into this axis; otherwise make a new figure.
    cmap : str
        Matplotlib colormap for continuous color_by.
    scatter_kwargs : dict
        Extra kwargs passed to the colored scatter (e.g., s=5, edgecolors='none').

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # --- Prepare embedding as DataFrame with two columns --- #
    if not isinstance(embedding, pd.DataFrame):
        embedding = pd.DataFrame(embedding)

    # Handle dims as names or indices
    if isinstance(dims[0], (int, np.integer)):
        cols = [embedding.columns[dims[0]], embedding.columns[dims[1]]]
    else:
        cols = list(dims)

    emb2d = embedding.loc[:, cols].copy()
    emb2d.columns = ["dim1", "dim2"]

    # --- Prepare annot --- #
    needed_cols = [color_by]
    if alpha_by is not None:
        needed_cols.append(alpha_by)

    # annot = annot.loc[emb2d.index, needed_cols].copy()
    annot = annot.reindex(emb2d.index)[needed_cols]

    df = pd.concat([emb2d, annot], axis=1)

    # --- Set up axis --- #
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Decide on rasterization for large datasets
    rasterized_flag = df.shape[0] > raster_thresh

    # --- Background grey points --- #
    ax.scatter(
        df["dim1"],
        df["dim2"],
        s=0.5,
        color="lightgrey",
        alpha=0.5,
        rasterized=rasterized_flag,
    )

    # --- Foreground colored points (non-NA color_by) --- #
    mask_color = df[color_by].notna()
    df_col = df.loc[mask_color].copy()

    # Build kwargs for alpha mapping
    scatter_args = dict(
        x=df_col["dim1"].values,
        y=df_col["dim2"].values,
    )

    # color mapping: if categorical, let matplotlib auto-assign; if numeric, use cmap
    color_vals = df_col[color_by]
    if pd.api.types.is_numeric_dtype(color_vals):
        scatter_args["c"] = color_vals.values
        scatter_args["cmap"] = cmap
    else:
        # treat as categorical; let matplotlib assign discrete colors
        cats = color_vals.astype("category")
        scatter_args["c"] = cats.cat.codes.values
        scatter_args["cmap"] = cmap

    if alpha_by is not None:
        scatter_args["alpha"] = df_col[alpha_by].values
    else:
        scatter_args["alpha"] = 1.0

    # Default point size
    if "s" not in scatter_kwargs:
        scatter_kwargs["s"] = 5.0

    ax.scatter(rasterized=rasterized_flag, **scatter_args, **scatter_kwargs)

    # --- Connectivity & labels --- #
    if label or (connectivity is not None):
        # center_coord: median coords & count per group
        center = (
            df.groupby(color_by, observed=True)
              .agg(
                  dim1=("dim1", "median"),
                  dim2=("dim2", "median"),
                  count=(color_by, "size"),
              )
              .dropna()
              .reset_index()
        )

        # Connectivity edges (if provided)
        if connectivity is not None and len(connectivity) > 0:
            edge_coord = connectivity.copy()
            edge_coord = edge_coord[edge_coord["x"] >= connectivity_thresh].copy()

            if not edge_coord.empty:
                ax.plot(
                    [edge_coord["i_x"], edge_coord["j_x"]],
                    [edge_coord["i_y"], edge_coord["j_y"]],
                    linewidth=edge_coord["x"],
                    color="honeydew",
                    alpha=0.75,
                    zorder=1,
                )

                # Draw cluster centers, sized by log(count)
                sizes = np.log(center["count"].values + 1.0) * 20.0
                ax.scatter(
                    center["dim1"],
                    center["dim2"],
                    s=sizes,
                    color="black",
                    alpha=0.8,
                    zorder=3,
                )

        # Labels
        if label and not center.empty:
            for _, row in center.iterrows():
                if label_type == "label":
                    ax.text(
                        row["dim1"],
                        row["dim2"],
                        str(row[color_by]),
                        fontsize=label_size,
                        color=label_color,
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad={}".format(box_padding),
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.8,
                        ),
                        zorder=4,
                    )
                else:  # "text"
                    ax.text(
                        row["dim1"],
                        row["dim2"],
                        str(row[color_by]),
                        fontsize=label_size,
                        color=label_color,
                        ha="center",
                        va="center",
                        zorder=4,
                    )

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_title(color_by)
    ax.set_aspect("equal")

    return fig, ax