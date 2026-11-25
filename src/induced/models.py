"""Models from the 2023 paper plus later additions.

All models are wrapped as imblearn Pipelines: scaler -> (resampler) -> clf, so
resampling only ever touches training folds.

Features are min-max scaled to [0, 1]; the paper's SVM settings (C=4, gamma=16)
only make sense on unit-scaled inputs.
"""
from __future__ import annotations

import numpy as np
from imblearn import combine, over_sampling as ov, under_sampling as un
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


# ---------------- Ordinal logistic (Frank & Hall 2001 decomposition) ----------------
class OrdinalLogit(BaseEstimator, ClassifierMixin):
    """Ordered classes 0<1<2: binary logistic models for P(y>0) and P(y>1),
    combined as P0=1-P(>0), P1=P(>0)-P(>1), P2=P(>1) (clipped and renormalised)."""

    def __init__(self, C=1.0, class_weight=None):
        self.C = C
        self.class_weight = class_weight

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(y, int)
        self.classes_ = np.arange(3)
        self.models_ = []
        for k in (0, 1):
            t = (y > k).astype(int)
            if t.min() == t.max():           # threshold never crossed in training
                self.models_.append(float(t.mean()))
            else:
                self.models_.append(LogisticRegression(C=self.C, class_weight=self.class_weight,
                                                       max_iter=5000).fit(X, t))
        return self

    def predict_proba(self, X):
        g = [np.full(len(X), m) if isinstance(m, float) else m.predict_proba(X)[:, 1] for m in self.models_]
        p_gt0, p_gt1 = g[0], np.minimum(g[1], g[0])
        p = np.c_[1 - p_gt0, p_gt0 - p_gt1, p_gt1].clip(0, 1)
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# ---------------- PNN (Specht 1990; Eqs. 22-23) ----------------
class PNN(BaseEstimator, ClassifierMixin):
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def fit(self, X, y):
        self.X_ = np.asarray(X, float)
        self.y_ = np.asarray(y, int)
        self.classes_ = np.unique(self.y_)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        d2 = ((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(-1)
        k = np.exp(-d2 / (2 * self.sigma ** 2))
        s = np.stack([k[:, self.y_ == c].mean(1) for c in self.classes_], 1)
        s = s + 1e-300
        return s / s.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


# ------------- AdaBoost (SAMME) with weighted resampling --------------
class AdaBoostResample(BaseEstimator, ClassifierMixin):
    """SAMME AdaBoost for base learners without sample_weight (e.g. KNN):
    each round fits on a bootstrap drawn with the current weights."""

    def __init__(self, base=None, n_estimators=100, random_state=0):
        self.base = base
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n = len(y)
        w = np.full(n, 1.0 / n)
        self.est_, self.alpha_ = [], []
        base = self.base if self.base is not None else KNeighborsClassifier(40)
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True, p=w)
            if len(np.unique(y[idx])) < K:
                idx = np.concatenate([idx, [np.where(y == c)[0][0] for c in self.classes_]])
            est = clone(base)
            if hasattr(est, "n_neighbors"):
                est.set_params(n_neighbors=min(est.n_neighbors, len(idx) - 1))
            est.fit(X[idx], y[idx])
            miss = est.predict(X) != y
            err = np.clip((w * miss).sum(), 1e-10, None)
            if err >= 1 - 1.0 / K:
                continue
            alpha = np.log((1 - err) / err) + np.log(K - 1)
            w = w * np.exp(alpha * miss)
            w /= w.sum()
            self.est_.append(est)
            self.alpha_.append(alpha)
        if not self.est_:
            self.est_, self.alpha_ = [clone(base).fit(X, y)], [1.0]
        return self

    def predict_proba(self, X):
        votes = np.zeros((len(X), len(self.classes_)))
        for est, a in zip(self.est_, self.alpha_):
            p = est.predict(X)
            votes[np.arange(len(X)), np.searchsorted(self.classes_, p)] += a
        votes = votes + 1e-9
        return votes / votes.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


# ---------------- Resamplers (Table 5) ----------------
def resamplers(seed=0):
    return {
        "None": None,
        "RandomOver": ov.RandomOverSampler(random_state=seed),
        "SMOTE": ov.SMOTE(random_state=seed),
        "BorderlineSMOTE": ov.BorderlineSMOTE(random_state=seed),
        "ADASYN": ov.ADASYN(random_state=seed),
        "SVMSMOTE": ov.SVMSMOTE(random_state=seed),
        "ClusterCentroids": un.ClusterCentroids(random_state=seed),
        "RandomUnder": un.RandomUnderSampler(random_state=seed),
        "NearMiss": un.NearMiss(),
        "ENN": un.EditedNearestNeighbours(),
        "NCR": un.NeighbourhoodCleaningRule(),
        "OSS": un.OneSidedSelection(random_state=seed),
        "CNN": un.CondensedNearestNeighbour(random_state=seed),
        "AllKNN": un.AllKNN(),
        "SMOTEENN": combine.SMOTEENN(random_state=seed),
        "SMOTETomek": combine.SMOTETomek(random_state=seed),
    }


def make(model, resampler=None, seed=0, **params):
    """Build a pipeline. model in {'svm', 'pnn', 'ada_knn', 'logreg', 'lgbm'}."""
    if model == "svm":
        clf = SVC(kernel=params.get("kernel", "rbf"), C=params.get("C", 4.0),
                  gamma=params.get("gamma", 16.0), degree=params.get("degree", 3),
                  coef0=params.get("coef0", 0.0), probability=params.get("probability", True),
                  class_weight=params.get("class_weight"), random_state=seed)
    elif model == "pnn":
        clf = PNN(sigma=params.get("sigma", 0.1))
    elif model == "ada_knn":
        clf = AdaBoostResample(KNeighborsClassifier(params.get("k", 40)),
                               n_estimators=params.get("n_estimators", 100), random_state=seed)
    elif model == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=params.get("C", 1.0), class_weight=params.get("class_weight"),
                                 max_iter=5000)
    elif model == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=params.get("n_estimators", 200),
                             learning_rate=params.get("learning_rate", 0.05),
                             num_leaves=params.get("num_leaves", 7),
                             min_child_samples=params.get("min_child_samples", 10),
                             class_weight=params.get("class_weight", "balanced"),
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=seed, verbose=-1)
    elif model == "ordinal":
        clf = OrdinalLogit(C=params.get("C", 1.0), class_weight=params.get("class_weight"))
    elif model == "tabpfn":
        # pretrained tabular foundation model (Hollmann et al. 2025); no tuning
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device="cpu", random_state=seed,
                               n_estimators=params.get("n_estimators", 8))
    else:
        raise ValueError(model)
    steps = [("scale", MinMaxScaler())]
    if resampler is not None:
        steps.append(("resample", resampler))
    steps.append(("clf", clf))
    return Pipeline(steps)
