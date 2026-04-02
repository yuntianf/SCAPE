import numpy as np


def aipw_ite_ow(Y, T, y0_hat, y1_hat, e_hat, eps=1e-3):
    Y      = np.asarray(Y, dtype=np.float32).reshape(-1, 1)
    y0_hat = np.asarray(y0_hat, dtype=np.float32).reshape(-1, 1)
    y1_hat = np.asarray(y1_hat, dtype=np.float32).reshape(-1, 1)
    T      = np.asarray(T, dtype=np.float32).reshape(-1, 1)
    e      = np.asarray(e_hat, dtype=np.float32).reshape(-1, 1)

    n, k = Y.shape
#     if Y.shape != y0_hat.shape or Y.shape != y1_hat.shape:
#         raise ValueError("Y, y0_hat, y1_hat must have same shape.")
    if T.shape[0] != n or e.shape[0] != n:
        raise ValueError("Incompatible n between Y, T, e_hat.")

    # Trim on e
    mask = ((e >= eps) & (e <= 1.0 - eps)).flatten()
    if mask.sum() == 0:
        raise ValueError("No samples after trimming; increase eps or check e_hat.")

    Ym   = Y[mask, :]
    y0m  = y0_hat[mask, :]
    y1m  = y1_hat[mask, :]
    Tm   = T[mask]
    em   = e[mask]
    m    = mask.sum()

    # Overlap weights within trimmed set
    w = em * (1.0 - em)              # (m,1)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("Sum of overlap weights is non-positive; check e_hat.")

    # Core DR terms
    term1 = y1m - y0m
    term2 = (Tm / em) * (Ym - y1m)
    term3 = ((1.0 - Tm) / (1.0 - em)) * (Ym - y0m)

    phi   = term1 + term2 - term3    # (m,k)

    # Construct OW pseudo-outcomes whose mean is tau_hat
    alpha  = w / w_sum                       # (m,1), sum alpha = 1
    tau_ow = (m * alpha) * phi               # (m,k), mean = tau_hat

    return tau_ow


def _to_1d_float(x, name):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D after reshape.")
    return x


def _validate_inputs(y, t, e, clip=1e-6):
    y = _to_1d_float(y, "y")
    t = _to_1d_float(t, "t")
    e = _to_1d_float(e, "e")

    if not (len(y) == len(t) == len(e)):
        raise ValueError("y, t, and e must have the same length.")

    if not np.all((t == 0) | (t == 1)):
        raise ValueError("t must be binary with values 0/1.")

    e = np.clip(e, clip, 1.0 - clip)
    return y, t, e


def ipw_ate(y, t, e, clip=1e-6, return_details=False):
    """
    Horvitz-Thompson IPW estimator for ATE.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed outcomes.
    t : array-like, shape (n,)
        Treatment indicators (0/1).
    e : array-like, shape (n,)
        Estimated propensity scores P(T=1|X).
    clip : float
        Clip propensity into [clip, 1-clip] for numerical stability.
    return_details : bool
        Whether to return weights and component estimates.

    Returns
    -------
    ate : float
        IPW estimate of ATE.
    details : dict, optional
    """
    y, t, e = _validate_inputs(y, t, e, clip=clip)

    w_t = t / e
    w_c = (1.0 - t) / (1.0 - e)

    mu1 = np.mean(w_t * y)
    mu0 = np.mean(w_c * y)
    ate = mu1 - mu0

    if return_details:
        return ate, {
            "mu1_hat": mu1,
            "mu0_hat": mu0,
            "w_treated": w_t,
            "w_control": w_c,
            "propensity_clipped": e,
        }
    return ate


def stabilized_ipw_ate(y, t, e, clip=1e-6, use_sample_marginal=True, p_treat=None, return_details=False):
    """
    Stabilized / Hajek IPW estimator for ATE.

    Two equivalent common variants are supported:
    1) normalized Hajek form (default behavior in the returned ATE),
    2) stabilized weights using marginal treatment probability.

    The final ATE returned here is:
        sum(w1*y)/sum(w1) - sum(w0*y)/sum(w0)

    where
        w1 = T * p / e
        w0 = (1-T) * (1-p) / (1-e)

    If use_sample_marginal=True, p is estimated as mean(T).
    Otherwise pass p_treat explicitly.

    Parameters
    ----------
    y, t, e : arrays
    clip : float
    use_sample_marginal : bool
    p_treat : float or None
    return_details : bool

    Returns
    -------
    ate : float
    details : dict, optional
    """
    y, t, e = _validate_inputs(y, t, e, clip=clip)

    if use_sample_marginal:
        p = float(np.mean(t))
    else:
        if p_treat is None:
            raise ValueError("Provide p_treat if use_sample_marginal=False.")
        p = float(p_treat)

    if not (0.0 < p < 1.0):
        raise ValueError("p_treat must be in (0, 1).")

    w1 = t * (p / e)
    w0 = (1.0 - t) * ((1.0 - p) / (1.0 - e))

    denom1 = np.sum(w1)
    denom0 = np.sum(w0)
    if denom1 <= 0 or denom0 <= 0:
        raise ValueError("Degenerate weights: denominator is zero.")

    mu1 = np.sum(w1 * y) / denom1
    mu0 = np.sum(w0 * y) / denom0
    ate = mu1 - mu0

    if return_details:
        return ate, {
            "mu1_hat": mu1,
            "mu0_hat": mu0,
            "w_treated": w1,
            "w_control": w0,
            "p_treat": p,
            "propensity_clipped": e,
        }
    return ate


def aipw_ate(y, t, e, m0, m1, clip=1e-6, return_details=False):
    """
    AIPW / DR estimator for ATE.

    Parameters
    ----------
    y : observed outcomes
    t : treatment 0/1
    e : estimated propensity
    m0 : predicted E[Y|T=0,X]
    m1 : predicted E[Y|T=1,X]

    Returns
    -------
    ate : float
    """
    y, t, e = _validate_inputs(y, t, e, clip=clip)
    m0 = _to_1d_float(m0, "m0")
    m1 = _to_1d_float(m1, "m1")

    if not (len(m0) == len(y) == len(m1)):
        raise ValueError("m0 and m1 must have the same length as y.")

    psi1 = m1 + t * (y - m1) / e
    psi0 = m0 + (1.0 - t) * (y - m0) / (1.0 - e)

    ate = np.mean(psi1 - psi0)

    if return_details:
        return ate, {
            "pseudo_outcome_treated": psi1,
            "pseudo_outcome_control": psi0,
            "propensity_clipped": e,
        }
    return ate