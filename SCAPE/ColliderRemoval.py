import copy

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Dict
from scipy.spatial.distance import cdist


# ============================================================
# Config
# ============================================================
@dataclass
class JointEntropicConfig:
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    epsilon: float = 0.02
    beta_treated: float = 1.10

    n_iters: int = 5000
    lr: float = 5e-2
    grad_clip: Optional[float] = 10.0
    verbose_every: int = 200
    seed: int = 0

    lambda_l2: float = 1e-8
    clamp_logits: float = 50.0

    # new
    tol_col: float = 1e-5
    tol_row: float = 1e-5
    stable_patience: int = 30
    lr_patience: int = 100
    lr_decay: float = 0.5
    min_lr: float = 1e-4
    stop_patience: int = 400
        
        
# ============================================================
# Joint dual model
# ============================================================

class JointEntropicDualOT(torch.nn.Module):
    """
    Solve the entropic joint OT problem:

        min_{P_u, P_t >= 0}
            <C_u, P_u> + beta <C_t, P_t>
            + eps * sum P_u (log P_u - 1)
            + eps * sum P_t (log P_t - 1)

        s.t.
            P_u^T 1 = u
            P_t^T 1 = t
            P_u 1 + P_t 1 <= a

    Dual variables:
      alpha_u  : column equality for untreated
      alpha_t  : column equality for treated
      lam >= 0 : shared row-capacity for A

    Optimal primal plans are:
      P_u[i,j] = exp((alpha_u[j] - lam[i] - C_u[i,j]) / eps)
      P_t[i,k] = exp((alpha_t[k] - lam[i] - beta*C_t[i,k]) / eps)
    """

    def __init__(
        self,
        C_u,
        C_t,
        a,
        u,
        t,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg

        self.C_u = torch.as_tensor(C_u, dtype=cfg.dtype, device=cfg.device)
        self.C_t = torch.as_tensor(C_t, dtype=cfg.dtype, device=cfg.device)
        self.a = torch.as_tensor(a, dtype=cfg.dtype, device=cfg.device)
        self.u = torch.as_tensor(u, dtype=cfg.dtype, device=cfg.device)
        self.t = torch.as_tensor(t, dtype=cfg.dtype, device=cfg.device)

        n_A, n_U = self.C_u.shape
        n_A2, n_T = self.C_t.shape
        if n_A != n_A2:
            raise ValueError("C_u and C_t must share the same number of rows (A).")

        self.alpha_u = torch.nn.Parameter(torch.zeros(n_U, dtype=cfg.dtype, device=cfg.device))
        self.alpha_t = torch.nn.Parameter(torch.zeros(n_T, dtype=cfg.dtype, device=cfg.device))
        self.raw_lam = torch.nn.Parameter(torch.zeros(n_A, dtype=cfg.dtype, device=cfg.device))

    def lam(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_lam)

    def primal_plans(self):
        """
        Recover primal plans from current dual variables.
        """
        eps = self.cfg.epsilon
        lam = self.lam()

        alpha_u = torch.clamp(self.alpha_u, -self.cfg.clamp_logits, self.cfg.clamp_logits)
        alpha_t = torch.clamp(self.alpha_t, -self.cfg.clamp_logits, self.cfg.clamp_logits)
        lam = torch.clamp(lam, 0.0, self.cfg.clamp_logits)

        logP_u = (alpha_u[None, :] - lam[:, None] - self.C_u) / eps
        logP_t = (alpha_t[None, :] - lam[:, None] - self.cfg.beta_treated * self.C_t) / eps

        P_u = torch.exp(logP_u)
        P_t = torch.exp(logP_t)
        return P_u, P_t

    def dual_objective(self):
        """
        Concave dual objective to maximize:

            D(alpha_u, alpha_t, lam)
              = alpha_u·u + alpha_t·t - lam·a
                - eps * sum exp((alpha_u - lam - C_u)/eps)
                - eps * sum exp((alpha_t - lam - beta C_t)/eps)

        We return:
          - dual_obj (to maximize)
          - diagnostics
        """
        eps = float(self.cfg.epsilon)
        lam = self.lam()

        alpha_u = torch.clamp(self.alpha_u, -self.cfg.clamp_logits, self.cfg.clamp_logits)
        alpha_t = torch.clamp(self.alpha_t, -self.cfg.clamp_logits, self.cfg.clamp_logits)
        lam = torch.clamp(lam, 0.0, self.cfg.clamp_logits)

        Z_u = (alpha_u[None, :] - lam[:, None] - self.C_u) / eps
        Z_t = (alpha_t[None, :] - lam[:, None] - self.cfg.beta_treated * self.C_t) / eps

        exp_u = torch.exp(Z_u)
        exp_t = torch.exp(Z_t)

        dual = (
            torch.dot(alpha_u, self.u)
            + torch.dot(alpha_t, self.t)
            - torch.dot(lam, self.a)
            - eps * exp_u.sum()
            - eps * exp_t.sum()
        )

        # Tiny L2 regularization for numerical stability only
        if self.cfg.lambda_l2 > 0:
            dual = dual - self.cfg.lambda_l2 * (
                (alpha_u ** 2).sum() + (alpha_t ** 2).sum() + (lam ** 2).sum()
            )

        # Diagnostics from primal reconstruction
        P_u = exp_u
        P_t = exp_t
        row_u = P_u.sum(dim=1)
        row_t = P_t.sum(dim=1)
        col_u = P_u.sum(dim=0)
        col_t = P_t.sum(dim=0)
        row_total = row_u + row_t
        excess = torch.clamp(row_total - self.a, min=0.0)

        diag = {
            "dual": dual.detach(),
            "col_err_u_l1": torch.abs(col_u - self.u).sum().detach(),
            "col_err_t_l1": torch.abs(col_t - self.t).sum().detach(),
            "over_capacity_l1": excess.sum().detach(),
            "mass_u": P_u.sum().detach(),
            "mass_t": P_t.sum().detach(),
            "mean_overlap": torch.minimum(row_u, row_t).mean().detach(),
        }
        return dual, diag
    
    
def fit_jot(
    X: np.ndarray,
    is_post: np.ndarray,
    is_treated: np.ndarray,
    weights: Optional[np.ndarray] = None,
    cross_cost_quantile = 0.5,
    cfg: JointEntropicConfig = JointEntropicConfig(),
) -> Dict[str, np.ndarray]:
    X = _as_float64(X)
    is_post = np.asarray(is_post).astype(bool)
    is_treated = np.asarray(is_treated).astype(bool)

    pre_mask = ~is_post
    post_untreated_mask = is_post & (~is_treated)
    post_treated_mask = is_post & is_treated

    idx_pre = np.where(pre_mask)[0]
    idx_u = np.where(post_untreated_mask)[0]
    idx_t = np.where(post_treated_mask)[0]
    
    X_work = X.copy()

    X_pre = X_work[idx_pre]
    X_u = X_work[idx_u]
    X_t = X_work[idx_t]

    # Uniform masses by default
    if weights is not None:
        a_u = weights[post_untreated_mask]
        a_t = weights[post_treated_mask]
        b_pre = weights[pre_mask]
        
        a_total = a_u.sum()+a_t.sum()
        a_u = a_u/total
        a_t = a_t/total
        
        b_pre = b_pre/b_pre.sum()
        
    else:
        n_u = X_u.shape[0]
        n_t = X_t.shape[0]
        n_pre = X_pre.shape[0]

        total_post = n_u + n_t

        a_u = np.ones(X_u.shape[0], dtype=np.float64) / total_post
        a_t = np.ones(X_t.shape[0], dtype=np.float64) / total_post
        b_pre = np.ones(X_pre.shape[0], dtype=np.float64) / n_pre
       
    M_u = pairwise_sqeuclidean(X_u, X_pre)
    stage1_scale = _robust_cost_scale(M_u, quantile=cross_cost_quantile)
    M_u = M_u / stage1_scale

    M_t = pairwise_sqeuclidean(X_t, X_pre)
    stage2_scale = _robust_cost_scale(M_t, quantile=cross_cost_quantile)
    M_t = M_t / stage2_scale
    
    model = JointEntropicDualOT(C_u=M_u.T, C_t=M_t.T, a=b_pre, u=a_u, t=a_t, cfg=cfg).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    
    # --- scheduler / early-stop hyperparameters
    tol_col = getattr(cfg, "tol_col", 1e-5)
    tol_row = getattr(cfg, "tol_row", 1e-5)

    # stop after residuals are all below tolerance for this many consecutive rounds
    stable_patience = getattr(cfg, "stable_patience", 30)

    # plateau scheduler on best score
    lr_patience = getattr(cfg, "lr_patience", 100)
    lr_decay = getattr(cfg, "lr_decay", 0.5)
    min_lr = getattr(cfg, "min_lr", 1e-4)

    # hard stop if no score improvement for too long
    stop_patience = getattr(cfg, "stop_patience", 400)

    history = []

    best_score = float("inf")
    best_dual = -float("inf")
    best_iter = 0
    best_state = None

    stable_rounds = 0
    no_improve_rounds = 0
    no_improve_rounds_lr = 0

    final_iter = 0

    for it in range(1, cfg.n_iters + 1):
        final_iter = it

        opt.zero_grad()
        dual, diag = model.dual_objective()
        loss = -dual
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        opt.step()

        row = {k: float(v.detach().cpu()) for k, v in diag.items()}
        row["iter"] = it
        row["lr"] = float(opt.param_groups[0]["lr"])

        # main residual score for checkpointing / scheduling
        score = row["col_err_u_l1"] + row["col_err_t_l1"] + row["over_capacity_l1"]
        row["score"] = score
        history.append(row)

        # best checkpoint: primarily by residual score, break ties by larger dual
        improved = False
        if score < best_score - 1e-10:
            improved = True
        elif abs(score - best_score) <= 1e-10 and row["dual"] > best_dual + 1e-12:
            improved = True

        if improved:
            best_score = score
            best_dual = row["dual"]
            best_iter = it
            best_state = copy.deepcopy(model.state_dict())
            no_improve_rounds = 0
            no_improve_rounds_lr = 0
        else:
            no_improve_rounds += 1
            no_improve_rounds_lr += 1

        # practical convergence: residuals all small for several consecutive rounds
        if (
            row["col_err_u_l1"] < tol_col
            and row["col_err_t_l1"] < tol_col
            and row["over_capacity_l1"] < tol_row
        ):
            stable_rounds += 1
        else:
            stable_rounds = 0

        # anneal LR on plateau
        if no_improve_rounds_lr >= lr_patience:
            old_lr = opt.param_groups[0]["lr"]
            new_lr = max(old_lr * lr_decay, min_lr)

            if new_lr < old_lr - 1e-15:
                # restore best checkpoint before continuing
                if best_state is not None:
                    model.load_state_dict(best_state)

                # re-create optimizer to clear Adam momentum
                opt = torch.optim.Adam(model.parameters(), lr=new_lr)

                if cfg.verbose_every:
                    print(
                        f"[iter {it:04d}] plateau/degrade detected: "
                        f"restore best_iter={best_iter}, "
                        f"reduce lr {old_lr:.3e} -> {new_lr:.3e}, "
                        f"reset optimizer state"
                    )

            no_improve_rounds_lr = 0

        # logging
        if cfg.verbose_every and (it == 1 or it % cfg.verbose_every == 0):
            print(
                f"[iter {it:04d}] "
                f"dual={row['dual']:.6f} "
                f"col_u={row['col_err_u_l1']:.3e} "
                f"col_t={row['col_err_t_l1']:.3e} "
                f"overcap={row['over_capacity_l1']:.3e} "
                f"score={row['score']:.3e} "
                f"lr={row['lr']:.3e} "
                f"best_iter={best_iter}"
            )

        # early stop: stable near-feasible solution
        if stable_rounds >= stable_patience:
            if cfg.verbose_every:
                print(
                    f"Early stop at iter {it}: residuals below tolerance for "
                    f"{stable_patience} consecutive rounds."
                )
            break

        # early stop: nothing improves for a long time and LR already tiny
        if no_improve_rounds >= stop_patience and opt.param_groups[0]["lr"] <= min_lr + 1e-15:
            if cfg.verbose_every:
                print(
                    f"Early stop at iter {it}: no improvement for {stop_patience} rounds "
                    f"at min_lr."
                )
            break

    # restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        
    with torch.no_grad():
        P_u, P_t = model.primal_plans()

        # Optional renormalization to remove tiny residual numerical mismatch
        # in column sums due to finite optimization.
        col_u = P_u.sum(dim=0, keepdim=True)
        col_t = P_t.sum(dim=0, keepdim=True)
        
        P_u = P_u * (model.u[None, :] / torch.clamp(col_u, min=1e-12))
        P_t = P_t * (model.t[None, :] / torch.clamp(col_t, min=1e-12))

        # After column correction, row inequality may be very slightly violated.
        # You can inspect usage and, if needed, increase n_iters or lower epsilon.
        usage_u = P_u.sum(dim=1)
        usage_t = P_t.sum(dim=1)

    P_u = P_u.cpu().numpy().T
    P_t = P_t.cpu().numpy().T
    
    n_pre = (~is_post).sum()
    n_post = (is_post).sum()
    
    t_post = is_treated[is_post]

    idx_u_local = np.where(t_post == 0)[0]
    idx_t_local = np.where(t_post == 1)[0]
    
    P_full = np.zeros((n_post,n_pre), dtype=np.float64)
    P_full[idx_u_local, :] = P_u
    P_full[idx_t_local, :] = P_t

    return {
        "P_u": P_u,
        "P_t": P_t,
        "P_full": P_full,
        "usage_u": usage_u.cpu().numpy(),
        "usage_t": usage_t.cpu().numpy(),
        "history": history,
    }


def _as_float64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)

def _normalize_histogram(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, eps)
    s = w.sum()
    if s <= 0:
        raise ValueError("Histogram has non-positive total mass.")
    return w / s


def _robust_cost_scale(M: np.ndarray, quantile: float = 0.5, eps: float = 1e-12) -> float:
    """Median or chosen quantile of positive entries for robust normalization."""
    vals = M[np.isfinite(M)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    scale = float(np.quantile(vals, quantile))
    return max(scale, eps)


def pairwise_sqeuclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return cdist(X, Y, metric="sqeuclidean")

def row_mass_sparsify(P, keep_mass=0.8, eps=1e-12):
    """
    Sparsify a coupling matrix P by keeping the largest entries in each row
    whose cumulative mass reaches 'keep_mass' (e.g. 0.8), then renormalize.

    Parameters
    ----------
    P : array (n_src, n_tgt)
        OT coupling matrix (non-negative).
    keep_mass : float in (0,1]
        Fraction of row mass to retain.
    eps : float
        Numerical stability constant.

    Returns
    -------
    P_sparse : array (n_src, n_tgt)
        Row-sparsified and renormalized coupling.
    """

    P = np.asarray(P, dtype=np.float64)
    n_src, n_tgt = P.shape

    P_sparse = np.zeros_like(P)

    for i in range(n_src):
        row = P[i]

        if row.sum() <= eps:
            continue  # leave as zero row (rare edge case)

        # Sort indices by descending weight
        idx_sorted = np.argsort(-row)

        # Cumulative mass
        cum_mass = np.cumsum(row[idx_sorted])
        total_mass = cum_mass[-1]

        # Find cutoff index
        cutoff = np.searchsorted(cum_mass, keep_mass * total_mass)

        # Indices to keep
        keep_idx = idx_sorted[:cutoff + 1]

        # Keep and renormalize
        kept_values = row[keep_idx]
        kept_values = kept_values / (kept_values.sum() + eps)

        P_sparse[i, keep_idx] = kept_values

    return P_sparse
