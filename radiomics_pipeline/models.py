"""Model zoo, hyper-parameter search spaces, leakage-free pipelines and search strategies
(grid, random, successive halving, Optuna TPE)."""
from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC

from .selection import SELECTORS_WITH_K, CorrelationFilter, DropConstant, logistic, make_selector

log = logging.getLogger(__name__)

# search-space spec: list -> categorical; ("float"|"int", low, high, log) -> range
F = lambda lo, hi, log=False: ("float", lo, hi, log)  # noqa: E731
I = lambda lo, hi, log=False: ("int", lo, hi, log)    # noqa: E731


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def model_zoo(seed: int, balanced: bool, n_classes: int, pos_weight: float = 1.0) -> dict:
    """name -> (estimator, search space)."""
    cw = "balanced" if balanced else None
    zoo = {
        "logistic": (logistic(l1_ratio=0.5, solver="saga", max_iter=5000, class_weight=cw, random_state=seed),
                     {"C": F(1e-3, 1e2, True), "l1_ratio": [0.0, 0.5, 1.0]}),
        "svm": (SVC(probability=True, class_weight=cw, random_state=seed),
                {"C": F(1e-2, 1e2, True), "gamma": F(1e-4, 1.0, True), "kernel": ["rbf", "linear"]}),
        "random_forest": (RandomForestClassifier(class_weight=cw, random_state=seed, n_jobs=1),
                          {"n_estimators": I(100, 600), "max_depth": [None, 3, 5, 8],
                           "min_samples_leaf": I(1, 10), "max_features": ["sqrt", "log2", 0.3]}),
        "extra_trees": (ExtraTreesClassifier(class_weight=cw, random_state=seed, n_jobs=1),
                        {"n_estimators": I(100, 600), "max_depth": [None, 3, 5, 8],
                         "min_samples_leaf": I(1, 10), "max_features": ["sqrt", "log2", 0.3]}),
        "gradient_boosting": (GradientBoostingClassifier(random_state=seed),
                              {"n_estimators": I(50, 400), "learning_rate": F(0.01, 0.3, True),
                               "max_depth": I(2, 5), "subsample": F(0.6, 1.0)}),
        "adaboost": (AdaBoostClassifier(random_state=seed),
                     {"n_estimators": I(50, 400), "learning_rate": F(0.01, 1.0, True)}),
        "knn": (KNeighborsClassifier(),
                {"n_neighbors": I(3, 25), "weights": ["uniform", "distance"], "p": [1, 2]}),
        "naive_bayes": (GaussianNB(), {"var_smoothing": F(1e-11, 1e-5, True)}),
        "lda": (LinearDiscriminantAnalysis(solver="lsqr"), {"shrinkage": ["auto", 0.0, 0.25, 0.5, 0.75]}),
        "mlp": (MLPClassifier(max_iter=2000, early_stopping=False, random_state=seed),
                {"hidden_layer_sizes": [(16,), (32,), (64,), (32, 16)], "alpha": F(1e-5, 1.0, True),
                 "learning_rate_init": F(1e-4, 1e-2, True)}),
    }
    if _has("xgboost"):
        from xgboost import XGBClassifier

        kw = {"scale_pos_weight": pos_weight} if balanced and n_classes == 2 else {}
        zoo["xgboost"] = (XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss", verbosity=0, **kw),
                          {"n_estimators": I(50, 500), "max_depth": I(2, 6), "learning_rate": F(0.01, 0.3, True),
                           "subsample": F(0.6, 1.0), "colsample_bytree": F(0.5, 1.0),
                           "min_child_weight": I(1, 10), "reg_lambda": F(1e-3, 10.0, True)})
    if _has("lightgbm"):
        from lightgbm import LGBMClassifier

        zoo["lightgbm"] = (LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, class_weight=cw),
                           {"n_estimators": I(50, 500), "num_leaves": I(4, 31), "learning_rate": F(0.01, 0.3, True),
                            "min_child_samples": I(3, 30), "subsample": F(0.6, 1.0), "subsample_freq": [1],
                            "colsample_bytree": F(0.5, 1.0), "reg_lambda": F(1e-3, 10.0, True)})
    if _has("catboost"):
        from catboost import CatBoostClassifier

        zoo["catboost"] = (CatBoostClassifier(random_seed=seed, verbose=0, thread_count=1,
                                              auto_class_weights="Balanced" if balanced else None),
                           {"iterations": I(100, 600), "depth": I(2, 6), "learning_rate": F(0.01, 0.3, True),
                            "l2_leaf_reg": F(1.0, 10.0, True)})
    return zoo


SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler,
           "quantile": lambda: QuantileTransformer(output_distribution="normal", n_quantiles=50)}


class SafeSampler(BaseEstimator):
    """Wrap an imbalanced-learn sampler; return the data unchanged when the sampler cannot
    resample (fold already balanced, or too few minority samples for the neighbours)."""

    def __init__(self, sampler=None):
        self.sampler = sampler

    def fit_resample(self, X, y, **kw):
        y = np.asarray(y)
        counts = np.bincount(y)
        try:
            if counts.min() == counts.max():
                return X, y
            s = clone(self.sampler)
            k = getattr(s, "k_neighbors", getattr(s, "n_neighbors", None))
            if isinstance(k, int) and counts[counts > 0].min() <= k:
                s.set_params(**{"k_neighbors" if hasattr(s, "k_neighbors") else "n_neighbors":
                                max(1, int(counts[counts > 0].min()) - 1)})
            return s.fit_resample(X, y)
        except (ValueError, RuntimeError) as e:
            log.debug("sampler skipped: %s", e)
            return X, y

    def fit(self, X, y):
        return self


def _sampler(name: str, seed: int):
    if name in ("none", "class_weight", None):
        return None
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    return SafeSampler({"smote": SMOTE(random_state=seed, k_neighbors=3),
                        "adasyn": ADASYN(random_state=seed, n_neighbors=3),
                        "undersample": RandomUnderSampler(random_state=seed)}[name])


def build_pipeline(selector: str, model: str, cfg: dict, y) -> tuple[object, dict]:
    """Return (pipeline, search space with 'clf__'/'selector__' prefixes)."""
    m = cfg["modeling"]
    seed = cfg["seed"]
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    imb = m.get("imbalance", "class_weight")
    pos_w = float((y == 0).sum() / max(1, (y == 1).sum())) if n_classes == 2 else 1.0
    zoo = model_zoo(seed, imb == "class_weight", n_classes, pos_w)
    if model not in zoo:
        raise ValueError(f"unknown/unavailable model {model!r}; available: {sorted(zoo)}")
    est, space = zoo[model]
    k_grid = list(m.get("selector_k") or [10])
    steps = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("constant", DropConstant()),
        ("scale", SCALERS[m.get("scaler", "standard")]()),
        ("decorrelate", CorrelationFilter(m.get("correlation_threshold", 0.9))),
        ("selector", make_selector(selector, k=k_grid[0], seed=seed)),
    ]
    sampler = _sampler(imb, seed)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", clone(est)))
    if sampler is not None:
        from imblearn.pipeline import Pipeline
    else:
        from sklearn.pipeline import Pipeline
    pipe = Pipeline(steps)
    full = {f"clf__{k}": v for k, v in space.items()}
    if selector.lower() in SELECTORS_WITH_K and len(k_grid) > 1:
        full["selector__k"] = k_grid
    return pipe, full


# --------------------------------------------------------------------------- search strategies
def to_grid(space: dict, points: int = 3) -> dict:
    g = {}
    for k, v in space.items():
        if isinstance(v, list):
            g[k] = v
        else:
            kind, lo, hi, lg = v
            vals = np.geomspace(lo, hi, points) if lg else np.linspace(lo, hi, points)
            g[k] = sorted({int(round(x)) for x in vals}) if kind == "int" else [float(x) for x in vals]
    return g


def to_distributions(space: dict) -> dict:
    d = {}
    for k, v in space.items():
        if isinstance(v, list):
            d[k] = v
        else:
            kind, lo, hi, lg = v
            if kind == "int":
                d[k] = randint(lo, hi + 1) if not lg else [int(x) for x in np.unique(np.geomspace(lo, hi, 10).round())]
            else:
                d[k] = loguniform(lo, hi) if lg else uniform(lo, hi - lo)
    return d


class OptunaSearchCV(BaseEstimator, ClassifierMixin):
    """Bayesian (TPE) hyper-parameter search with the same surface as sklearn's *SearchCV."""

    def __init__(self, estimator=None, space=None, n_trials=30, cv=3, scoring="roc_auc", random_state=0,
                 timeout=None):
        self.estimator, self.space, self.n_trials, self.cv = estimator, space, n_trials, cv
        self.scoring, self.random_state, self.timeout = scoring, random_state, timeout

    def _suggest(self, trial):
        params = {}
        for k, v in self.space.items():
            if isinstance(v, list):
                idx = trial.suggest_categorical(k, list(range(len(v))))  # index -> supports tuples/None
                params[k] = v[idx]
            else:
                kind, lo, hi, lg = v
                params[k] = (trial.suggest_int(k, lo, hi, log=lg) if kind == "int"
                             else trial.suggest_float(k, lo, hi, log=lg))
        return params

    def fit(self, X, y, groups=None):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            est = clone(self.estimator).set_params(**self._suggest(trial))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring, groups=groups,
                                    error_score=np.nan)
            return float(np.nanmean(s)) if np.isfinite(s).any() else -1e9

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.study_ = study
        self.best_params_ = self._suggest(optuna.trial.FixedTrial(study.best_params))
        self.best_score_ = study.best_value
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)
        self.classes_ = self.best_estimator_.classes_
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)


def make_search(pipe, space: dict, cfg: dict, cv, scoring: str, n_samples: int | None = None):
    m = cfg["modeling"]
    kind = m.get("search", "random")
    seed = cfg["seed"]
    n_iter = int(m.get("n_iter", 20))
    if not space:
        return GridSearchCV(pipe, {}, cv=cv, scoring=scoring, error_score=np.nan)
    if kind == "grid":
        return GridSearchCV(pipe, to_grid(space), cv=cv, scoring=scoring, error_score=np.nan, n_jobs=1)
    if kind == "random":
        return RandomizedSearchCV(pipe, to_distributions(space), n_iter=n_iter, cv=cv, scoring=scoring,
                                  random_state=seed, error_score=np.nan, n_jobs=1)
    if kind == "halving":
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401
        from sklearn.model_selection import HalvingRandomSearchCV

        # small radiomics cohorts: never train on fewer than ~30 patients in the first rung
        n = int(n_samples or 60)
        return HalvingRandomSearchCV(pipe, to_distributions(space), n_candidates=3 * n_iter, cv=cv, scoring=scoring,
                                     random_state=seed, error_score=np.nan, factor=2,
                                     min_resources=int(min(n, max(30, n // 4))))
    if kind == "optuna":
        return OptunaSearchCV(pipe, space, n_trials=n_iter, cv=cv, scoring=scoring, random_state=seed)
    raise ValueError(f"unknown search {kind!r}")
