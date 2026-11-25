"""Phase 0: rebuild the 2023 dataset, reproduce Tables 3/5, then test a
chronological split.

Run from followup/src:
    python ../scripts/phase0_reproduce.py
Outputs go to $IEQ_WORK/results/phase0/.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from induced import catalog, config, features as F, injection, metrics, models, splits

OUT = config.RESULTS / "phase0"
OUT.mkdir(parents=True, exist_ok=True)
N_SEEDS = 50


def build():
    path = config.PROCESSED / "orig_dataset.parquet"
    cat = catalog.load()
    inj = injection.swd_only(injection.build())
    wells = pd.read_parquet(config.PROCESSED / "wells_central_swd.parquet")
    df = F.build_dataset(cat, inj, wells, config.CENTRAL_OK, config.ORIG_START, config.ORIG_END)
    df.to_parquet(path)
    return df


def fit_eval(df, tr, te, model, rs_name, seed, **params):
    X, y = df[F.FEATURES].values, df["y"].values
    rs = models.resamplers(seed)[rs_name]
    pipe = models.make(model, rs, seed=seed, **params)
    try:
        pipe.fit(X[tr], y[tr])
    except Exception:
        return np.nan, None
    pred = pipe.predict(X[te])
    return metrics.mava(y[te], pred), pred


MODELS = {
    "SVM_RBF": ("svm", dict(kernel="rbf", C=4, gamma=16, probability=False)),
    "PNN": ("pnn", dict(sigma=0.1)),
    "AdaBoost_KNN": ("ada_knn", dict(k=40, n_estimators=100)),
}
KERNELS = {
    "linear": dict(kernel="linear", C=4),
    "poly": dict(kernel="poly", C=4, degree=3, gamma=4, coef0=5),
    "rbf": dict(kernel="rbf", C=4, gamma=16),
    "sigmoid": dict(kernel="sigmoid", C=0.0625, gamma=4, coef0=2),
}


def main(df=None, out=None):
    """df: optional prebuilt dataset (e.g. the original 2023 input matrix); out: output folder."""
    global OUT
    if out is not None:
        OUT = Path(out)
        OUT.mkdir(parents=True, exist_ok=True)
    df = build() if df is None else df
    df = df.dropna(subset=F.FEATURES).reset_index(drop=True)
    print("dataset", df.shape, "classes", np.bincount(df.y))
    df[["start", "mmax", "y"] + F.FEATURES].to_csv(OUT / "dataset_original_window.csv", index=False)

    # ---- Table 5: resampler x model, random 75/25 over many seeds ----
    rows = []
    for seed in range(N_SEEDS):
        (tr, te), = splits.random_split(df, 0.25, seed=seed)
        for mname, (m, p) in MODELS.items():
            for rs in models.resamplers().keys():
                s, _ = fit_eval(df, tr, te, m, rs, seed, **p)
                rows.append(dict(seed=seed, model=mname, resampler=rs, mava=s))
    t5 = pd.DataFrame(rows)
    t5.to_csv(OUT / "table5_random_all_seeds.csv", index=False)
    summ = t5.groupby(["resampler", "model"]).mava.agg(["mean", "std", "max"]).unstack("model")
    summ.round(3).to_csv(OUT / "table5_random_summary.csv")
    print(summ.round(2))

    # ---- Table 3: kernels with SMOTE ----
    rows = []
    for seed in range(N_SEEDS):
        (tr, te), = splits.random_split(df, 0.25, seed=seed)
        for k, p in KERNELS.items():
            s, _ = fit_eval(df, tr, te, "svm", "SMOTE", seed, probability=False, **p)
            rows.append(dict(seed=seed, kernel=k, mava=s))
    t3 = pd.DataFrame(rows).groupby("kernel").mava.agg(["mean", "std", "max"])
    t3.round(3).to_csv(OUT / "table3_kernels_random.csv")
    print(t3.round(2))

    # ---- Chronological split: same models, same settings ----
    rows = []
    for gap in (0, 18):
        (tr, te), = splits.chrono_split(df, 0.25, gap=gap)
        for mname, (m, p) in MODELS.items():
            for rs in models.resamplers().keys():
                vals = [fit_eval(df, tr, te, m, rs, s, **p)[0] for s in range(5)]
                rows.append(dict(gap=gap, model=mname, resampler=rs, mava=np.nanmean(vals)))
    ch = pd.DataFrame(rows)
    ch.to_csv(OUT / "table5_chronological.csv", index=False)
    print(ch[ch.gap == 0].pivot(index="resampler", columns="model", values="mava").round(2))

    # ---- Headline: SVM-RBF + SMOTE, random vs chronological, with CIs ----
    head = []
    for label, folds in [("random_seed0", splits.random_split(df, 0.25, seed=0)),
                         ("chrono", splits.chrono_split(df, 0.25)),
                         ("chrono_gap9mo", splits.chrono_split(df, 0.25, gap=18))]:
        tr, te = folds[0]
        _, pred = fit_eval(df, tr, te, "svm", "SMOTE", 0, kernel="rbf", C=4, gamma=16, probability=False)
        s = metrics.summary(df.y.values[te], pred)
        s.update(split=label, test_from=str(df.start.iloc[te].min().date()),
                 test_to=str(df.start.iloc[te].max().date()))
        head.append(s)
    rnd = t5[(t5.model == "SVM_RBF") & (t5.resampler == "SMOTE")].mava
    head.append(dict(split=f"random_{N_SEEDS}_seeds_mean", mava=rnd.mean(),
                     mava_lo=rnd.quantile(.025), mava_hi=rnd.quantile(.975)))
    hd = pd.DataFrame(head)
    hd.to_csv(OUT / "headline_svm_smote.csv", index=False)
    print(hd.round(3).to_string())


if __name__ == "__main__":
    main()
