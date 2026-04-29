# SCAPE

**S**patial **C**ausal **A**nalysis with **P**aired **E**mbeddings

SCAPE is a Python toolkit for causal effect estimation in spatial transcriptomics. It targets a specific challenge in paired pre/post-treatment tissue designs: observed cell-state features (e.g. gene expression) contain *colliders* — variables jointly caused by both the outcome and the treatment — which bias standard causal estimators. SCAPE removes this bias via joint entropic optimal transport and combines the result with doubly robust AIPW estimation.

---

## Key ideas

In a paired spatial experiment, cells are measured before and after treatment on the same tissue. The observed feature matrix **K** contains:

- **U** — true confounders that drive both treatment assignment and outcome
- **C** — collider features (e.g. expression programs activated by treatment *and* outcome) that introduce spurious correlations in the post-period

Naively regressing on **K** conflates U and C. SCAPE uses joint entropic OT to couple each post-period cell to its most plausible pre-period counterpart, effectively conditioning on the pre-treatment state and breaking the collider path.

---

## Modules

| Module | Description |
|---|---|
| `ColliderRemoval` | Joint entropic OT (`fit_jot`) for collider bias removal |
| `OTSample` | Stochastic sampling from OT coupling matrices |
| `CausalEffect` | IPW, stabilized IPW, and AIPW-OW point estimators |
| `CausalRegression` | Cross-fitted AIPW with Ridge / logistic nuisance models |
| `DragonNet` | Multi-output DragonNet with adversarial slide/pre-post heads and cross-fitted AIPW |
| `NeighborCount` | Slide-aware spatial neighbor-type count and treatment proximity matrices |
| `NeighborEmbedding` | Pairwise cell-neighborhood distances via graph geodesics (NN-surrogate or exact EMD) |
| `Preprocess` | Per-gene z-scaling, neighbor-count feature transformation, and slide one-hot encoding |
| `BuildGraph` | Connected SNN hybrid graph construction for geodesic distance computation |
| `BatchRemovalHarmony` | Harmony-based batch correction for multi-slide data |
| `BatchRemovalSymphony` | Symphony-style reference-build + query-map batch correction |
| `Simulation` | Synthetic data generation with configurable confounding, colliders, and batch effects |
| `Visualization` | Scatter plots, UMAP embeddings, vector fields, and counterfactual outcome plots |

---

## Installation

```bash
git clone https://github.com/<your-username>/SCAPE.git
cd SCAPE
pip install -e .
```

**Dependencies** (installed automatically):

```
numpy, scipy, pandas, scikit-learn, torch, harmonypy, umap-learn, igraph, leidenalg
```

---

## Quick start

```python
import SCAPE
import numpy as np
from sklearn.decomposition import PCA

# 1. Simulate paired pre/post data with collider confounding
data = SCAPE.simulate_observed_confounder(
    n_slides_pre=3, n_slides_post=3,
    tau=0.5, alpha_y=2, alpha_t=4,
    U_simulator=SCAPE.continuity_generator,
    add_batch=False, seed=42,
)

# 2. Embed observed features (PCA fit on untreated cells)
from sklearn.decomposition import PCA
K     = (data["K"] - data["K"].mean(0)) / data["K"].std(0).clip(1e-8)
pca   = PCA().fit(K[data["T"] == 0])
K_pca = pca.transform(K)

# 3. Fit joint entropic OT to remove collider bias
cfg    = SCAPE.JointEntropicConfig(epsilon=0.01, beta_treated=0.1)
result = SCAPE.fit_jot(X=K_pca, is_post=data["is_post"], is_treated=data["T"], cfg=cfg)

# 4. Sample corrected embeddings and estimate ATE via AIPW
is_post = data["is_post"] == 1
mapping = SCAPE.row_mass_sparsify(result["P_full"], keep_mass=0.9)
samples = SCAPE.sample_map_projection(mapping, K_pca[~is_post], n_samples=60)

ATE_list = []
for X_corr in samples:
    res = SCAPE.aipw_ate_crossfit(X_corr, data["T"][is_post], data["Y"][is_post].reshape(-1,1))
    ATE_list.append(res["ate"])

print(f"Estimated ATE: {np.mean(ATE_list):.3f}  (true tau = 0.5)")
```

See **`tutorial.ipynb`** for a full annotated walkthrough.

---

## Tutorial

`tutorial.ipynb` covers the complete pipeline end-to-end:

1. Data simulation with ground-truth confounding
2. Baseline causal estimate on raw features (shows bias)
3. Joint entropic OT collider removal
4. Propensity score recovery diagnostics
5. ATE estimation via cross-fitted AIPW
6. ITE estimation via OT coupling

---

## License

[MIT](LICENSE)
