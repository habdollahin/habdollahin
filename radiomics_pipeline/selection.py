"""Feature-selection transformers (scikit-learn API). They are placed *inside* the modelling
pipeline so they are re-fitted on each training fold only (no selection leakage).

Available (``modeling.selectors``):
  none, variance, anova, mannwhitney, mutual_info, fdr, mrmr, lasso, elasticnet,
  rfe_svm, rfe_lr, rf_importance, extratrees, boruta, relieff
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

log = logging.getLogger(__name__)


class _Selector(BaseEstimator, TransformerMixin):
    """Base: subclasses set ``support_`` (bool mask) in ``fit``."""

    def transform(self, X):
        check_is_fitted(self, "support_")
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices=False):
        return np.flatnonzero(self.support_) if indices else self.support_

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features if input_features is not None
                           else getattr(self, "feature_names_in_", np.arange(len(self.support_))))
        return names[self.support_]

    def _topk(self, scores, k):
        scores = np.nan_to_num(np.asarray(scores, float), nan=-np.inf)
        k = int(min(max(1, k), len(scores)))
        sup = np.zeros(len(scores), bool)
        sup[np.argsort(-scores, kind="stable")[:k]] = True
        return sup


class DropConstant(_Selector):
    """Remove (quasi-)constant features: variance <= threshold or one value in >= max_freq of samples."""

    def __init__(self, threshold=1e-8, max_freq=0.95):
        self.threshold, self.max_freq = threshold, max_freq

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        var = np.nanvar(X, axis=0)
        freq = np.array([np.unique(c[~np.isnan(c)], return_counts=True)[1].max() / max(1, (~np.isnan(c)).sum())
                         if (~np.isnan(c)).any() else 1.0 for c in X.T])
        self.support_ = (var > self.threshold) & (freq < self.max_freq)
        if not self.support_.any():
            self.support_ = var > 0
        return self


class CorrelationFilter(_Selector):
    """Greedy removal of highly inter-correlated features (|Spearman| > threshold). From each
    correlated pair the feature with lower univariate relevance (ANOVA F to y) is dropped."""

    def __init__(self, threshold=0.9, method="spearman"):
        self.threshold, self.method = threshold, method

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        p = X.shape[1]
        if self.threshold is None or self.threshold >= 1 or p < 2:
            self.support_ = np.ones(p, bool)
            return self
        Z = stats.rankdata(X, axis=0) if self.method == "spearman" else X
        C = np.abs(np.nan_to_num(np.corrcoef(Z, rowvar=False)))
        rel = np.nan_to_num(f_classif(X, y)[0]) if y is not None else -C.mean(0)
        order = np.argsort(-rel, kind="stable")
        keep = np.zeros(p, bool)
        dropped = np.zeros(p, bool)
        for i in order:
            if dropped[i]:
                continue
            keep[i] = True
            dropped |= C[i] > self.threshold
        self.support_ = keep
        return self


class UnivariateSelector(_Selector):
    """Top-k by a univariate test: anova (F), mannwhitney / kruskal (H), mutual_info.
    With ``alpha`` set, keep features with Benjamini-Hochberg q < alpha (at least ``k_min``)."""

    def __init__(self, test="anova", k=10, alpha=None, k_min=2, random_state=0):
        self.test, self.k, self.alpha, self.k_min, self.random_state = test, k, alpha, k_min, random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        if self.test == "anova":
            score, p = f_classif(X, y)
        elif self.test == "mannwhitney":
            groups = [X[y == c] for c in np.unique(y)]
            if len(groups) == 2:
                res = stats.mannwhitneyu(groups[0], groups[1], axis=0)
            else:
                res = stats.kruskal(*groups, axis=0)
            p = res.pvalue
            score = -np.log10(np.clip(p, 1e-300, 1))
        elif self.test == "mutual_info":
            score = mutual_info_classif(X, y, random_state=self.random_state)
            p = None
        else:
            raise ValueError(self.test)
        self.scores_ = np.nan_to_num(score)
        if self.alpha is not None and p is not None:
            q = benjamini_hochberg(np.nan_to_num(p, nan=1.0))
            self.support_ = q < self.alpha
            if self.support_.sum() < self.k_min:
                self.support_ = self._topk(self.scores_, self.k_min)
        else:
            self.support_ = self._topk(self.scores_, self.k)
        return self


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


class MRMR(_Selector):
    """Minimum-Redundancy Maximum-Relevance (MID variant): relevance = mutual information with y,
    redundancy = mean |Pearson r| with already selected features."""

    def __init__(self, k=10, random_state=0):
        self.k, self.random_state = k, random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        p = X.shape[1]
        k = int(min(self.k, p))
        rel = mutual_info_classif(X, y, random_state=self.random_state)
        C = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
        if C.ndim == 0:
            C = np.ones((1, 1))
        selected = [int(np.argmax(rel))]
        red = C[selected[0]].copy()
        while len(selected) < k:
            score = rel - red / len(selected)
            score[selected] = -np.inf
            j = int(np.argmax(score))
            selected.append(j)
            red += C[j]
        self.support_ = np.zeros(p, bool)
        self.support_[selected] = True
        self.ranking_ = selected
        return self


class ModelSelector(_Selector):
    """Keep the top-k features by |coef| or feature_importances_ of a fitted model. For L1 models
    with ``k=None`` all non-zero coefficients are kept."""

    def __init__(self, estimator=None, k=10):
        self.estimator, self.k = estimator, k

    def fit(self, X, y):
        est = clone(self.estimator).fit(X, y)
        if hasattr(est, "coef_"):
            imp = np.abs(np.atleast_2d(est.coef_)).sum(0)
        else:
            imp = est.feature_importances_
        self.importances_ = imp
        if self.k is None:
            self.support_ = imp > 1e-10
            if self.support_.sum() == 0:
                self.support_ = self._topk(imp, 1)
        else:
            nz = int((imp > 1e-10).sum())
            self.support_ = self._topk(imp, min(self.k, nz) if nz else self.k)
        return self


class RFESelector(_Selector):
    def __init__(self, estimator=None, k=10, step=0.1):
        self.estimator, self.k, self.step = estimator, k, step

    def fit(self, X, y):
        k = int(min(self.k, np.asarray(X).shape[1]))
        r = RFE(clone(self.estimator), n_features_to_select=k, step=self.step).fit(X, y)
        self.support_ = r.support_
        self.ranking_ = r.ranking_
        return self


class Boruta(_Selector):
    """All-relevant selection (Kursa & Rudnicki 2010): features are compared with shuffled
    'shadow' copies over many random-forest fits; a two-sided binomial test decides
    confirmed / rejected. Tentative features are resolved by median importance vs best shadow.
    ``k`` caps the number kept (by mean importance)."""

    def __init__(self, n_iter=50, alpha=0.05, k=None, n_estimators=100, random_state=0):
        self.n_iter, self.alpha, self.k, self.n_estimators, self.random_state = n_iter, alpha, k, n_estimators, random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)
        hits = np.zeros(p)
        imps = []
        for it in range(self.n_iter):
            shadow = X.copy()
            for j in range(p):
                shadow[:, j] = rng.permutation(shadow[:, j])
            rf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=5, class_weight="balanced",
                                        n_jobs=1, random_state=int(rng.integers(1 << 31)))
            rf.fit(np.hstack([X, shadow]), y)
            imp = rf.feature_importances_
            hits += imp[:p] > imp[p:].max()
            imps.append(imp[:p] - np.median(imp[p:]))
        imps = np.array(imps)
        p_hi = stats.binom.sf(hits - 1, self.n_iter, 0.5)
        p_lo = stats.binom.cdf(hits, self.n_iter, 0.5)
        confirmed = p_hi < self.alpha
        rejected = p_lo < self.alpha
        tentative = ~confirmed & ~rejected
        confirmed |= tentative & (np.median(imps, 0) > 0)
        self.importances_ = imps.mean(0)
        self.hits_ = hits
        sup = confirmed
        if sup.sum() == 0:
            sup = self._topk(self.importances_, 3)
        if self.k is not None and sup.sum() > self.k:
            sc = np.where(sup, self.importances_, -np.inf)
            sup = self._topk(sc, self.k)
        self.support_ = sup
        return self


class ReliefF(_Selector):
    """ReliefF (Kononenko 1994) with k nearest hits/misses on min-max scaled features."""

    def __init__(self, k=10, n_neighbors=10):
        self.k, self.n_neighbors = k, n_neighbors

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        lo, hi = X.min(0), X.max(0)
        Z = (X - lo) / np.where(hi > lo, hi - lo, 1)
        n, p = Z.shape
        D = np.abs(Z[:, None, :] - Z[None, :, :]).sum(-1)
        np.fill_diagonal(D, np.inf)
        classes, counts = np.unique(y, return_counts=True)
        prior = dict(zip(classes, counts / n))
        w = np.zeros(p)
        for i in range(n):
            for c in classes:
                idx = np.flatnonzero(y == c)
                idx = idx[idx != i]
                if len(idx) == 0:
                    continue
                nn = idx[np.argsort(D[i, idx])[: self.n_neighbors]]
                diff = np.abs(Z[nn] - Z[i]).mean(0)
                if c == y[i]:
                    w -= diff
                else:
                    w += prior[c] / (1 - prior[y[i]]) * diff
        self.scores_ = w / n
        self.support_ = self._topk(self.scores_, self.k)
        return self


def make_selector(name: str, k: int = 10, seed: int = 0):
    """Factory; ``k`` can be tuned via the ``selector__k`` hyper-parameter."""
    name = name.lower()
    if name in ("none", "all"):
        return "passthrough"
    if name == "variance":
        return DropConstant()
    if name in ("anova", "mannwhitney", "mutual_info"):
        return UnivariateSelector(name, k=k, random_state=seed)
    if name == "fdr":
        return UnivariateSelector("mannwhitney", k=k, alpha=0.05, k_min=2)
    if name == "mrmr":
        return MRMR(k=k, random_state=seed)
    if name == "lasso":
        return ModelSelector(logistic_cv(l1_ratio=1.0, Cs=10, cv=3, solver="saga", class_weight="balanced",
                                         max_iter=5000, random_state=seed), k=k)
    if name == "elasticnet":
        return ModelSelector(logistic(l1_ratio=0.5, C=0.5, solver="saga", class_weight="balanced",
                                      max_iter=5000, random_state=seed), k=k)
    if name == "rfe_svm":
        return RFESelector(LinearSVC(C=0.1, class_weight="balanced", max_iter=20000, random_state=seed), k=k)
    if name == "rfe_lr":
        return RFESelector(logistic(C=0.5, class_weight="balanced", max_iter=5000), k=k)
    if name == "rf_importance":
        return ModelSelector(RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed,
                                                    n_jobs=1), k=k)
    if name == "extratrees":
        return ModelSelector(ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=seed,
                                                  n_jobs=1), k=k)
    if name == "boruta":
        return Boruta(n_iter=30, k=k, random_state=seed)
    if name == "relieff":
        return ReliefF(k=k)
    raise ValueError(f"unknown selector {name!r}")


def _new_lr_api() -> bool:
    """scikit-learn >= 1.8 deprecates ``penalty`` in favour of ``l1_ratio``."""
    import inspect

    return inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated"


def logistic(l1_ratio: float = 0.0, **kw) -> LogisticRegression:
    """Version-independent LogisticRegression with an L1/L2/elastic-net mix."""
    if _new_lr_api():
        return LogisticRegression(l1_ratio=l1_ratio, **kw)
    pen = "l2" if l1_ratio == 0 else "l1" if l1_ratio == 1 else "elasticnet"
    extra = {"l1_ratio": l1_ratio} if pen == "elasticnet" else {}
    return LogisticRegression(penalty=pen, **extra, **kw)


def logistic_cv(l1_ratio: float = 1.0, **kw) -> LogisticRegressionCV:
    if _new_lr_api():
        return LogisticRegressionCV(l1_ratios=(l1_ratio,), scoring="neg_log_loss", use_legacy_attributes=False, **kw)
    pen = "l2" if l1_ratio == 0 else "l1" if l1_ratio == 1 else "elasticnet"
    extra = {"l1_ratios": [l1_ratio]} if pen == "elasticnet" else {}
    return LogisticRegressionCV(penalty=pen, scoring="neg_log_loss", **extra, **kw)


SELECTORS_WITH_K = {"anova", "mannwhitney", "mutual_info", "mrmr", "lasso", "elasticnet", "rfe_svm",
                    "rfe_lr", "rf_importance", "extratrees", "boruta", "relieff", "fdr"}
