import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LogisticRegression
from .CausalEffect import aipw_ite_ow


def aipw_ate_crossfit(X, T, Y, n_splits=5,
                      ridge_alpha_y=1.0,
                      ridge_C_t=1.0,
                      clip_e=(1e-3, 1-1e-3),
                      random_state=0):
    """
    AIPW estimator for ATE with cross-fitting.
    Regression-based: Ridge for outcomes, LogisticRegression (L2) for propensity.
    """
    X = np.asarray(X, float)
    T = np.asarray(T).astype(int)
    Y = np.asarray(Y, float)

    n = X.shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    m1_hat = np.zeros(n)
    m0_hat = np.zeros(n)
    e_hat  = np.zeros(n)

    for train, test in kf.split(X):
        Xtr, Xte = X[train], X[test]
        Ttr, Tte = T[train], T[test]
        Ytr      = Y[train]

        # Propensity model: P(T=1|X)
        clf = LogisticRegression(
            l1_ratio=0, C=ridge_C_t, solver="lbfgs", max_iter=2000
        )
        clf.fit(Xtr, Ttr)
        e = clf.predict_proba(Xte)[:, 1]
        e = np.clip(e, clip_e[0], clip_e[1])
        e_hat[test] = e

        # Outcome models: E[Y|T=t,X]
        # Fit separate ridge models in treated and control
        rid1 = Ridge(alpha=ridge_alpha_y, fit_intercept=True)
        rid0 = Ridge(alpha=ridge_alpha_y, fit_intercept=True)

        idx1 = (Ttr == 1)
        idx0 = (Ttr == 0)

        # handle edge cases: if a fold has too few treated/control, fall back to pooled
        if idx1.sum() < 5 or idx0.sum() < 5:
            rid = Ridge(alpha=ridge_alpha_y, fit_intercept=True)
            rid.fit(Xtr, Ytr)
            m = rid.predict(Xte)
            m1_hat[test] = m
            m0_hat[test] = m
        else:
            rid1.fit(Xtr[idx1], Ytr[idx1])
            rid0.fit(Xtr[idx0], Ytr[idx0])
            m1_hat[test] = rid1.predict(Xte)
            m0_hat[test] = rid0.predict(Xte)

    # AIPW score
#     tau_i = (m1_hat - m0_hat
#              + T * (Y - m1_hat) / e_hat
#              - (1 - T) * (Y - m0_hat) / (1 - e_hat))
    tau_i = aipw_ite_ow(Y, T, m0_hat, m1_hat, e_hat)

    ate = tau_i.mean()
    se = tau_i.std(ddof=1) / np.sqrt(n)  # simple influence-function SE
    return {"ate": ate, "se": se, "ci95": (ate - 1.96*se, ate + 1.96*se),
            "e_hat": e_hat, "m1_hat": m1_hat, "m0_hat": m0_hat}