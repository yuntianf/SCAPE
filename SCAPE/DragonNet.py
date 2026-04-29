# ====== Imports ======
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from scipy.stats import norm
from typing import Literal
from scipy import sparse

from . import Preprocess as pre
from .utils import GradientReversal


# =========================================
# DragonNet with Adversarial Slide Head
# =========================================

class DragonNet(keras.Model):
    """
    Multi-output DragonNet with scalar-epsilon targeted regularization (TR) and control for additive batch effect across slides.

    Architecture (mirrors DragonNet):
      - Representation Z(X): depth_rep × Dense(width_rep, activation)
      - Outcome heads y0, y1 = m(Z,S): two hidden layers then Dense(k)
      - Propensity head g(X)=e(Z,S): single Dense(1, sigmoid) from Z (logistic on rep)
      - Adversarial slide head sees GRL(Z) and predicts S (to make Z slide-invariant).

    Loss (exact DragonNet logic):
      L = MSE( Y, Q_theta(T,X,S) )
          + alpha_ps * BCE( T, e_theta(X,S) )
          + beta_tr  * MSE( Y, Q_theta(T,X,S) + ε * H(T,X; e_theta) )
          + gamma * BCE(S,d(X))
          + weight decay

      where H(T,X;e) = T/e - (1-T)/(1-e)  (clever covariate), ε is a learned scalar.
    """
    def __init__(self,
                 p, k, n_slides,n_groups=None,pre_post = False,
                 depth_rep=3, width_rep=200, width_head=(100, 100),
                 activation="elu",
                 l2=1e-4, dropout=0.0,
                 alpha_ps=1.0, beta_tr=1.0, clip_epsilon=1e-3,
                 grl_lambda=1.0, gamma_batch=1.0, gamma_prepost = 1.0, 
                 name="dragonnet", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = int(k)
        self.n_slides = int(n_slides)
        self.n_groups = None if n_groups is None else int(n_groups)
        self.pre_post = bool(pre_post)
        self.alpha_ps = float(alpha_ps)
        self.beta_tr  = float(beta_tr)
        self.clip_epsilon = float(clip_epsilon)
        self.gamma_batch = float(gamma_batch)
        self.gamma_prepost = float(gamma_prepost)

        act = activation
        reg = regularizers.l2(l2) if (l2 and l2 > 0) else None

        # Shared representation on X
        self.rep_layers = []
        for _ in range(int(depth_rep)):
            self.rep_layers.append(layers.Dense(width_rep, activation=act, kernel_regularizer=reg))
            if dropout and dropout > 0:
                self.rep_layers.append(layers.Dropout(dropout))

        w1, w2 = width_head if isinstance(width_head, (list, tuple)) else (width_head, width_head)

        # Outcome heads
        self.y0_h1 = layers.Dense(w1, activation=act, kernel_regularizer=reg)
        self.y0_h2 = layers.Dense(w2, activation=act, kernel_regularizer=reg)
        self.y0_out = layers.Dense(self.k, kernel_initializer="he_normal", name="y0")

        self.y1_h1 = layers.Dense(w1, activation=act, kernel_regularizer=reg)
        self.y1_h2 = layers.Dense(w2, activation=act, kernel_regularizer=reg)
        self.y1_out = layers.Dense(self.k, kernel_initializer="he_normal", name="y1")

        # ------- Additive slide offset gamma(S) -------
        # Map one-hot S -> R^k with a linear layer (no bias) to produce per-gene offsets.
        if self.n_slides > 1:
            self.gamma_layer = layers.Dense(self.k, use_bias=False, name="gamma_S", kernel_regularizer=reg)
        else:
            self.gamma_layer = None
            
        # Propensity head
        self.e_h = layers.Dense(w1, activation=act, kernel_regularizer=reg)
        self.e_out = layers.Dense(1, activation="sigmoid", name="e")

        # Adversarial slide head for batch effect(GRL on Z)
        self._adv_enabled = (self.n_slides >= 2)
        if self._adv_enabled:
            self.grl = GradientReversal(lambd=grl_lambda)
            self.d_h = layers.Dense(w1, activation=act, kernel_regularizer=reg)
            # Use logits (no activation) + CE(from_logits=True) for stability
            self.d_out = layers.Dense(self.n_slides, activation=None, name="slide_logits")
            self._dom_out_dim = self.n_slides
        else:
            self.grl = None
            self.d_h = None
            self.d_out = None
            # zero out the weight so even if computed it's ignored
            self.gamma_batch = 0.0
            self._dom_out_dim = 1 

        # Adversarial slide head for pre and post treatment(GRL on Z)
        if self.pre_post:
            self.pgrl = GradientReversal(lambd=grl_lambda)
            self.pd_h = layers.Dense(w1, activation=act, kernel_regularizer=reg)
            # Use logits (no activation) + CE(from_logits=True) for stability
            self.pd_out = layers.Dense(2, activation=None, name="prepost_logits")
        else:
            self.pgrl = None
            self.pd_h = None
            self.pd_out = None
            # zero out the weight so even if computed it's ignored
            self.gamma_prepost = 0.0
        

        # Scalar epsilon for TR
        self.epsilon = self.add_weight(name = "epsilon_tr", shape=(), initializer="zeros", trainable=True, dtype=tf.float32)

        # Metrics
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_tracker  = keras.metrics.Mean(name="mse")
        self.tracker_tr   = keras.metrics.Mean(name="tr_mse")
        self.bce_tracker  = keras.metrics.Mean(name="bce")
        self.dom_tracker  = keras.metrics.Mean(name="dom_ce")
        self.pdom_tracker  = keras.metrics.Mean(name="pdom_ce")

    def _rep(self, X, training=False):
        h = X
        for lyr in self.rep_layers:
            if isinstance(lyr, layers.Dropout):
                h = lyr(h, training=training)
            else:
                h = lyr(h)
        return h

    def call(self, inputs, training=False):
        # inputs: (X, S_onehot)
        X, S, G, P = inputs
        Z = self._rep(X, training=training)

        Zy = tf.concat([Z, G], axis=1)   # outcome heads see [Z, G]
        Ze = tf.concat([Z, G], axis=1)   # propensity head sees [Z, G]

        if self.gamma_layer is not None:
            gamma = self.gamma_layer(S)
        else:
            gamma = 0.0
            
        # Outcome heads
        f0 = self.y0_out(self.y0_h2(self.y0_h1(Zy)))
        f1 = self.y1_out(self.y1_h2(self.y1_h1(Zy)))
        y0 = f0+gamma
        y1 = f1+gamma
        
        # Propensity head
        e  = self.e_out(self.e_h(Ze))
        
        # Domain head (adversarial, no S as input)
        if self._adv_enabled:
            d_logits = self.d_out(self.d_h(self.grl(Z)))
        else:
            n_cell = tf.shape(Z)[0]
            d_logits = tf.zeros((n_cell, self._dom_out_dim), dtype=tf.float32)

        if self.pre_post:
            pd_logits = self.pd_out(self.pd_h(self.pgrl(Z)))
        else:
            n_cell = tf.shape(Z)[0]
            pd_logits = tf.zeros((n_cell, 2), dtype=tf.float32)

        return y0, y1, e, d_logits,pd_logits, f0,f1

    @property
    def metrics(self):
        return [self.loss_tracker, self.mse_tracker, self.tracker_tr, self.bce_tracker, self.dom_tracker, self.pdom_tracker]

    # data: ((X,S), T), Y
    def train_step(self, data):
        (X, S, G, P), (T, Y) = data
        X = tf.cast(X, tf.float32); 
        S = tf.cast(S, tf.float32); 
        G = tf.cast(G, tf.float32);
        P = tf.cast(P, tf.float32);
        T = tf.cast(T, tf.float32); 
        Y = tf.cast(Y, tf.float32)

        with tf.GradientTape() as tape:
            y0, y1, e, d_logits,pd_logits,*_ = self((X, S, G, P), training=True)
            Q_f = T * y1 + (1.0 - T) * y0

            # mean-over-genes MSE (so scales match BCE/CE)
            mse_rows = tf.reduce_mean(tf.square(Y - Q_f), axis=1)
            mse_loss = tf.reduce_mean(mse_rows)

            bce_loss = tf.reduce_mean(keras.losses.binary_crossentropy(T, e))

            e_clip = tf.clip_by_value(e, self.clip_epsilon, 1.0 - self.clip_epsilon)
            H = T / (e_clip + 1e-8) - (1.0 - T) / (1.0 - e_clip + 1e-8)
            Q_tilt = Q_f + self.epsilon * H
            tr_rows = tf.reduce_mean(tf.square(Y - Q_tilt), axis=1)
            tr_loss = tf.reduce_mean(tr_rows)

            # Domain loss (only if enabled)
            if self._adv_enabled:
                # one-hot S, logits predictions
                dom_ce = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(S, d_logits, from_logits=True)
                )
                K = tf.cast(self._dom_out_dim, tf.float32)
                dom_ce = dom_ce / (tf.math.log(K + 1e-8))
            else:
                dom_ce = tf.constant(0.0, dtype=tf.float32)

            # Pre Post loss (only if enabled)
            if self.pre_post:
                # one-hot S, logits predictions
                pdom_ce = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(P, pd_logits, from_logits=True)
                )
                K = tf.cast(2, tf.float32)
                pdom_ce = pdom_ce / (tf.math.log(K + 1e-8))
            else:
                pdom_ce = tf.constant(0.0, dtype=tf.float32)

            reg_loss = tf.add_n(self.losses) if self.losses else 0.0
            total = mse_loss + self.alpha_ps * bce_loss + self.beta_tr * tr_loss + self.gamma_batch * dom_ce + self.gamma_prepost * pdom_ce + reg_loss

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(total)
        self.mse_tracker.update_state(mse_loss)
        self.tracker_tr.update_state(tr_loss)
        self.bce_tracker.update_state(bce_loss)
        self.dom_tracker.update_state(dom_ce)
        self.pdom_tracker.update_state(pdom_ce)

        return {"loss": self.loss_tracker.result(), "mse": self.mse_tracker.result(),
                "tr_mse": self.tracker_tr.result(), "bce": self.bce_tracker.result(),
                "dom_ce": self.dom_tracker.result(), "pdom_ce": self.pdom_tracker.result(), 
                "epsilon": self.epsilon}

    def test_step(self, data):
        (X, S, G, P), (T, Y) = data
        X = tf.cast(X, tf.float32); 
        S = tf.cast(S, tf.float32);
        G = tf.cast(G, tf.float32);
        P = tf.cast(P, tf.float32);
        T = tf.cast(T, tf.float32); 
        Y = tf.cast(Y, tf.float32);

        y0, y1, e, d_logits,pd_logits,*_ = self((X, S, G, P), training=False)
        Q_f = T * y1 + (1.0 - T) * y0

        mse = tf.reduce_mean(tf.reduce_mean(tf.square(Y - Q_f), axis=1))
        bce = tf.reduce_mean(keras.losses.binary_crossentropy(T, e))
        e_clip = tf.clip_by_value(e, self.clip_epsilon, 1.0 - self.clip_epsilon)
        H = T / (e_clip + 1e-8) - (1.0 - T) / (1.0 - e_clip + 1e-8)
        Q_tilt = Q_f + self.epsilon * H
        tr_mse = tf.reduce_mean(tf.reduce_mean(tf.square(Y - Q_tilt), axis=1))

        if self._adv_enabled:
            dom_ce = tf.reduce_mean(
                keras.losses.categorical_crossentropy(S, d_logits, from_logits=True)
            )
            K = tf.cast(self._dom_out_dim, tf.float32)
            dom_ce = dom_ce / (tf.math.log(K + 1e-8))
        else:
            dom_ce = tf.constant(0.0, dtype=tf.float32)

        if self.pre_post:
            pdom_ce = tf.reduce_mean(
                keras.losses.categorical_crossentropy(P, pd_logits, from_logits=True)
            )
            K = tf.cast(2, tf.float32)
            pdom_ce = pdom_ce / (tf.math.log(K + 1e-8))
        else:
            pdom_ce = tf.constant(0.0, dtype=tf.float32)
        
        reg_loss = tf.add_n(self.losses) if self.losses else 0.0

        total = mse + self.alpha_ps * bce + self.beta_tr * tr_mse + self.gamma_batch * dom_ce + self.gamma_prepost * pdom_ce + reg_loss
        return {"loss": total, "mse": mse,
                "tr_mse": tr_mse, "bce": bce, "dom_ce": dom_ce, "pdom_ce": pdom_ce, 
                "epsilon": self.epsilon}


def build_dragonnet(p, k, n_slides, n_groups=None, pre_post=False,lr=1e-3, **kw):
    model = DragonNet(p=p, k=k, n_slides=n_slides,n_groups=n_groups,pre_post = pre_post, **kw)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


# =========================
# AIPW utilities
# =========================
def aipw_ite(Y, T, y0_hat, y1_hat, e_hat, eps=1e-3):
    """AIPW ITEs per gene using model outputs (n, k)."""
    Y       = np.asarray(Y, dtype=np.float32)
    y0_hat  = np.asarray(y0_hat, dtype=np.float32)
    y1_hat  = np.asarray(y1_hat, dtype=np.float32)
    T       = np.asarray(T, dtype=np.float32).reshape(-1, 1)
    e       = np.asarray(e_hat, dtype=np.float32).reshape(-1, 1)

    if Y.shape != y0_hat.shape or Y.shape != y1_hat.shape:
        raise ValueError(f"Y, y0_hat, y1_hat shapes must match, got "
                         f"{Y.shape}, {y0_hat.shape}, {y1_hat.shape}")
    if Y.shape[0] != T.shape[0] or Y.shape[0] != e.shape[0]:
        raise ValueError(f"Incompatible n between Y ({Y.shape[0]}), "
                         f"T ({T.shape[0]}), e_hat ({e.shape[0]})")

    mask = ((e >= eps) & (e <= 1.0 - eps)).flatten()

    ite = np.full_like(Y, np.nan, dtype=np.float32)
    
    term1 = y1_hat[mask,:] - y0_hat[mask,:]
    term2 = (T[mask] / e[mask]) * (Y[mask,:] - y1_hat[mask,:])
    term3 = ((1.0 - T[mask]) / (1.0 - e[mask])) * (Y[mask,:] - y0_hat[mask,:])

    ite[mask] = term1 + term2 - term3
    
    tau_hat = ite[mask].mean(axis=0)
    if_contrib = ite - tau_hat[None, :]
    
    return tau_hat,if_contrib

def aipw_ite_ow(Y, T, y0_hat, y1_hat, e_hat, eps=1e-3):
    Y      = np.asarray(Y, dtype=np.float32)
    y0_hat = np.asarray(y0_hat, dtype=np.float32)
    y1_hat = np.asarray(y1_hat, dtype=np.float32)
    T      = np.asarray(T, dtype=np.float32).reshape(-1, 1)
    e      = np.asarray(e_hat, dtype=np.float32).reshape(-1, 1)

    n, k = Y.shape
    if Y.shape != y0_hat.shape or Y.shape != y1_hat.shape:
        raise ValueError("Y, y0_hat, y1_hat must have same shape.")
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
    tau_hat = (w * phi).sum(axis=0) / w.sum() 
    if_contrib = (w / w.mean()) * (phi - tau_hat[None, :])
#     alpha  = w / w_sum                       # (m,1), sum alpha = 1
#     tau_ow = (m * alpha) * phi               # (m,k), mean = tau_hat

    return tau_hat,if_contrib

def _benjamini_hochberg(p):
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    min_q = 1.0
    for i in range(m - 1, -1, -1):
        min_q = min(min_q, ranked[i] * m / (i + 1))
        q[i] = min_q
    out = np.empty_like(q)
    out[order] = q
    return np.clip(out, 0, 1)

def aipw_test_z(Y, T, y0_hat, y1_hat, e_hat, eps=1e-3, option = "ow"):
    Y = np.asarray(Y, dtype=np.float32)

    # T can be 1D; aipw_ite will reshape it appropriately
    T = np.asarray(T, dtype=np.float32)

    if option == "ow":
        ate,psi = aipw_ite_ow(Y, T, y0_hat, y1_hat, e_hat, eps=eps)
    else:
        ate,psi = aipw_ite(Y, T, y0_hat, y1_hat, e_hat, eps=eps)
        
    psi = psi[~np.isnan(psi).any(axis=1)]
    n, k = psi.shape

#     ate  = tau.mean(axis=0)
#     psi  = tau - ate[None, :]
    se   = np.sqrt(np.sum(psi**2, axis=0) / (n * (n - 1)))
    zstat = np.divide(ate, se, out=np.zeros_like(ate), where=se > 0)
    pvals = 2.0 * (1.0 - norm.cdf(np.abs(zstat)))
    qvals = _benjamini_hochberg(pvals)

    return {"ate": ate, "p": pvals, "q": qvals}
    

# ======================================================
# K-fold split: within-slide & stratified by treatment
# ======================================================

def _kfold_within_slide_stratified(slide_ids, T, K=5, seed=1):
    """Create K folds so each fold has cells from every slide; stratify by T within each slide."""
    rng = np.random.default_rng(seed)
    Tbin = (T.reshape(-1) > 0.5).astype(int)
    folds = [list() for _ in range(K)]
    for s in np.unique(slide_ids):
        idx_s = np.where(slide_ids == s)[0]
        t0 = idx_s[Tbin[idx_s] == 0].copy()
        t1 = idx_s[Tbin[idx_s] == 1].copy()
        rng.shuffle(t0); rng.shuffle(t1)
        parts0 = np.array_split(t0, K)
        parts1 = np.array_split(t1, K)
        for k in range(K):
            folds[k].append(np.concatenate([parts0[k], parts1[k]]))
    folds = [np.concatenate(chunk_lists) if len(chunk_lists) else np.array([], dtype=int)
             for chunk_lists in folds]
    for f in folds:
        rng.shuffle(f)
    return folds


# ======================================================
# Cross-fitting with slides (adversarial)
# ======================================================

def make_train_ds(X, S, G, P, T, Y, batch):
    ds = tf.data.Dataset.from_tensor_slices(((X.astype('float32'), S.astype('float32'),G.astype('float32'),P.astype('float32')),
                                             (T.astype('float32'), Y.astype('float32'))))
    ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch)  # don't fix last batch size here; TF handles None batch dim better with DS
    return ds.prefetch(tf.data.AUTOTUNE)


def crossfit_dragonnet(
    X, T, Y, mode:Literal["count", "embedding"],
    slide_ids=None,group_ids=None,prepost_ids = None,
    K=5, seed=42,
    model_builder=build_dragonnet,
    model_kwargs=None,
    fit_kwargs=None,
    return_original_units=True,
    aipw_clip=1e-6,
    collect_history=False
):
    """
    Cross-fitting for DragonNet with adversarial slide head.
    Inputs
      X : (n, p_masked) raw masked confounders
      T : (n, 1) or (n,) binary treatment
      Y : (n, k) raw outcomes (will be z-scored per fold)
      slide_ids : (n,) array-like slide IDs (strings or ints)
      group_ids : (n,) array-like group IDs (strings or ints)
      prepost_ids : (n,1) IDs to indicate if the treatment happened (strings or ints)
    """
    if model_kwargs is None:
        model_kwargs = dict(
            depth_rep=3, width_rep=200, width_head=(100, 100),
            activation="elu", l2=1e-4, dropout=0.0,
            alpha_ps=1.0, beta_tr=1.0, clip_epsilon=1e-3,
            grl_lambda=1.0, gamma_batch=1.0, gamma_prepost=1.0, 
            lr=1e-3
        )
    if fit_kwargs is None:
        fit_kwargs = dict(
            epochs=50, batch_size=512, verbose=1,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        )

    X = np.asarray(X)
    if sparse.issparse(Y):
        Y = Y.toarray().astype(np.float32)
    else:
        Y = np.asarray(Y, dtype=np.float32)
    T = np.asarray(T).reshape(-1, 1).astype(np.float32)
    n, k = Y.shape
    
    if slide_ids is None:
        slide_ids = np.ones(n)
    if group_ids is None:
        group_ids = np.ones(n)
    if prepost_ids is None:
        prepost_ids = np.ones(n)
    slide_ids = np.asarray(slide_ids)
    group_ids = np.asarray(group_ids)
    prepost_ids = np.asarray(prepost_ids)
    
   
    y0_oof = np.zeros((n, k), dtype=np.float32)
    y1_oof = np.zeros((n, k), dtype=np.float32)
    f0_oof = np.zeros((n, k), dtype=np.float32)
    f1_oof = np.zeros((n, k), dtype=np.float32)
    e_oof  = np.zeros((n, 1), dtype=np.float32)
    Yz_oof = np.zeros((n, k), dtype=np.float32)
    sd_per_sample = np.ones((n, k), dtype=np.float32)
    hist_rows = []

    folds = _kfold_within_slide_stratified(slide_ids, T, K=K, seed=seed)

    for fold_id, te_idx in enumerate(folds, 1):
        tr_idx = np.setdiff1d(np.arange(n), te_idx, assume_unique=True)

        # Fit transforms on TRAIN only
        yscaler = pre.GeneZScaler().fit(Y[tr_idx, :])
        senc    = pre.SlideOneHotEncoder().fit(slide_ids[tr_idx])
        genc    = pre.SlideOneHotEncoder().fit(group_ids[tr_idx])
        penc    = pre.SlideOneHotEncoder().fit(prepost_ids[tr_idx]) 

        if mode == "count":
            xform   = pre.NeighborXTransformer().fit(X[tr_idx, :])
            Xtr = xform.transform(X[tr_idx, :]).astype(np.float32)
            Xte = xform.transform(X[te_idx,  :]).astype(np.float32)
        else:
            Xtr = X[tr_idx, :].astype(np.float32)
            Xte = X[te_idx, :].astype(np.float32)
            
        Str = senc.transform(slide_ids[tr_idx]).astype(np.float32)
        Ste = senc.transform(slide_ids[te_idx]).astype(np.float32)

        Gtr = genc.transform(group_ids[tr_idx]).astype(np.float32)
        Gte = genc.transform(group_ids[te_idx]).astype(np.float32)

        Ptr = penc.transform(prepost_ids[tr_idx]).astype(np.float32)
        Pte = penc.transform(prepost_ids[te_idx]).astype(np.float32)
        
        Ytr_z = yscaler.transform(Y[tr_idx, :]).astype(np.float32)
        Yte_z = yscaler.transform(Y[te_idx,  :]).astype(np.float32)
        Ttr = T[tr_idx, :].astype(np.float32)
        Tte = T[te_idx,  :].astype(np.float32)
    
        n_slides_fold = Str.shape[1]          # one-hot columns
        n_slides_arg  = n_slides_fold if n_slides_fold >= 2 else 0

        n_groups_fold = Gtr.shape[1]
        n_groups_arg  = n_groups_fold if n_groups_fold >= 1 else None

        n_prepost_fold = Ptr.shape[1]
        n_prepost_arg  = n_prepost_fold if n_prepost_fold == 2 else None
        
        model = model_builder(p=Xtr.shape[1], k=Ytr_z.shape[1], n_slides=n_slides_arg,n_groups=n_groups_arg, pre_post = n_prepost_arg, **model_kwargs)

        train_ds = make_train_ds(Xtr, Str, Gtr, Ptr, Ttr, Ytr_z, fit_kwargs['batch_size'])
        val_ds   = make_train_ds(Xte, Ste, Gte, Pte, Tte, Yte_z, fit_kwargs['batch_size'])
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            **fit_kwargs
        )

        # OOF predictions
        y0_hat, y1_hat, e_hat, d_logits_hat,pd_logits_hat, f0_hat,f1_hat = model.predict((Xte, Ste,Gte,Pte), 
                                                                                 batch_size=fit_kwargs.get("batch_size", 512), verbose=0)

        y0_oof[te_idx, :] = y0_hat
        y1_oof[te_idx, :] = y1_hat
        f0_oof[te_idx, :] = f0_hat
        f1_oof[te_idx, :] = f1_hat
        e_oof[te_idx,  :] = e_hat
        Yz_oof[te_idx, :] = Yte_z
        sd_per_sample[te_idx, :] = yscaler.sd  # for back-transform

        if collect_history:
            hh = history.history
            n_ep = len(next(iter(hh.values())))
            for ep in range(n_ep):
                row = {"fold": fold_id, "epoch": ep + 1}
                for kk, vv in hh.items(): row[kk] = vv[ep]
                hist_rows.append(row)

    # AIPW in z-units
    # tau_dr_z = aipw_ite(Yz_oof, T, y0_oof, y1_oof, e_oof, eps=aipw_clip)
    # ate_z  = tau_dr_z.mean(axis=0)
    # psi_z  = tau_dr_z - ate_z[None, :]
    # se_z   = np.sqrt(np.sum(psi_z**2, axis=0) / (n * (n - 1)))
    # zstat  = np.divide(ate_z, se_z, out=np.zeros_like(ate_z), where=se_z > 0)
    # pvals  = 2.0 * (1.0 - norm.cdf(np.abs(zstat)))
    # qvals  = _benjamini_hochberg(pvals)

    out = {
        "y0_hat": f0_oof, "y1_hat": f1_oof, "e_hat_oof": e_oof,
        "m0_hat": y0_oof, "m1_hat": y1_oof, "y_z": Yz_oof
        # "tau_dr_z": tau_dr_z, "ate_z": ate_z, "se_z": se_z, "z": zstat, "p": pvals, "q": qvals
    }

    # if return_original_units:
    #     tau_dr_orig = tau_dr_z * sd_per_sample
    #     ate = tau_dr_orig.mean(axis=0)
    #     psi_orig = tau_dr_orig - ate[None, :]
    #     se = np.sqrt(np.sum(psi_orig**2, axis=0) / (n * (n - 1)))
    #     out.update({"tau_dr_orig": tau_dr_orig, "ate": ate, "se": se})

    if collect_history:
        try:
            import pandas as pd
            out["history_df"] = pd.DataFrame(hist_rows)
        except Exception:
            out["history_rows"] = hist_rows

    return out

