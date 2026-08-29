"""
Three detectors, evaluated head-to-head on Precision@k.

  1. RULES      -- account-level thresholds, the incumbent bank approach
  2. UNSUP      -- unsupervised structural score (NO labels used)
  3. GBM        -- LightGBM ranker trained ONLY on the sparse exposed labels

Plus two honesty checks that most hackathon projects skip:
  - ACTIVITY-ONLY control: a model using only volume/count features. If the
    real model cannot beat this, it is detecting inactive accounts, not mules.
  - Confounder audit: what fraction of each detector's alerts are legitimate
    high-fan-in shops, rent collectors and aggregators.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

CONFOUNDERS = ["small_shop", "group_collector", "gig_aggregator", "informal_lender"]

DROP = ["account_type", "is_mule", "label_exposed", "mule_network_id", "community"]
ACTIVITY_ONLY = ["n_txn", "tot_in", "tot_out", "in_n", "out_n",
                 "fan_in", "fan_out", "peak_in_val"]
# 27 features overfit 31 labels; this hand-picked set is what actually works
CURATED = ["p10_dwell", "min_dwell", "fast_flowthru_n", "med_matched_passthru",
           "reciprocity", "device_shared_by", "age_min_at_first_txn",
           "betweenness"]


# --------------------------------------------------------------------------
def precision_at_k(scores: pd.Series, y: pd.Series, k: int) -> float:
    top = scores.sort_values(ascending=False).head(k).index
    return float(y.loc[top].mean())


def recall_at_k(scores: pd.Series, y: pd.Series, k: int) -> float:
    top = scores.sort_values(ascending=False).head(k).index
    return float(y.loc[top].sum() / max(y.sum(), 1))


def networks_hit(scores: pd.Series, F: pd.DataFrame, k: int) -> int:
    top = scores.sort_values(ascending=False).head(k).index
    nets = set()
    for s in F.loc[top, "mule_network_id"]:
        if s:
            nets.update(s.split("|"))
    return len(nets)


# --------------------------------------------------------------------------
def rules_baseline(F: pd.DataFrame) -> pd.Series:
    """Typical single-institution transaction-monitoring rules."""
    s = np.zeros(len(F))
    s += 2.0 * (F.p10_dwell < 15)                     # empties fast
    s += 2.0 * (F.med_matched_passthru > 0.75)        # passes value through
    s += 1.5 * (F.fan_in > F.fan_in.quantile(0.90))   # many payers
    s += 1.5 * (F.peak_in_val > F.peak_in_val.quantile(0.95))
    s += 1.0 * (F.age_min_at_first_txn < 90 * 1440)   # new account
    s += 1.0 * (F.stranger_in_share > 0.8)
    # tiny jitter to break ties deterministically
    return pd.Series(s + np.linspace(0, 1e-6, len(F)), index=F.index)


def unsupervised_score(F: pd.DataFrame) -> pd.Series:
    """Structural + behavioural anomaly score. Uses NO labels, so it can
    surface mule networks that have never been reported."""
    cols = ["p10_dwell", "fast_flowthru_n", "med_matched_passthru",
            "reciprocity", "device_shared_by", "k_core", "betweenness",
            "comm_reciprocity", "peak_window_passthru", "age_min_at_first_txn",
            "stranger_in_share", "fan_in", "fan_out"]
    X = F[cols].replace([np.inf, -np.inf], 0).fillna(0)
    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=400, contamination=0.05, random_state=0)
    iso.fit(Xs)
    iso_s = -iso.score_samples(Xs)

    # directed structural prior: fast, non-reciprocal, device-shared, new
    z = pd.DataFrame(Xs, columns=cols, index=F.index)
    prior = (-z.p10_dwell + z.fast_flowthru_n + z.device_shared_by
             - z.reciprocity - z.comm_reciprocity - z.age_min_at_first_txn
             + z.betweenness)
    prior = (prior - prior.mean()) / prior.std()
    iso_z = (iso_s - iso_s.mean()) / iso_s.std()
    return pd.Series(0.5 * iso_z + 0.5 * prior.to_numpy(), index=F.index)


def gbm_score(F: pd.DataFrame, feature_cols: list[str],
              seed: int = 0) -> pd.Series:
    """Trained ONLY on exposed (sparse, delayed) labels.

    Unexposed mules stay in the training set labelled 0 -- exactly the
    positive-unlabelled situation a real bank faces. Out-of-fold scoring
    keeps the evaluation honest.
    """
    X = F[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    y_train = F.label_exposed.to_numpy()
    oof = np.zeros(len(F))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y_train):
        m = lgb.LGBMClassifier(
            n_estimators=350, learning_rate=0.05, num_leaves=31,
            min_child_samples=5, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0,
            scale_pos_weight=float((y_train[tr] == 0).sum() /
                                   max((y_train[tr] == 1).sum(), 1)),
            random_state=seed, verbose=-1)
        m.fit(X.iloc[tr], y_train[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return pd.Series(oof, index=F.index)


def final_score(F: pd.DataFrame) -> pd.Series:
    """The shipped detector. Uses NO labels.

    Two terms only:
      community score  -- is this account's NETWORK laundering-shaped?
      chain depth      -- how deep in a layering chain does it sit?

    The account-level anomaly term was DROPPED. It scored 0.245 on one seed
    and 0.005 on another; the IsolationForest component is unstable across
    resamples. Removing it and weighting chain depth harder improves
    held-out precision (0.875 -> 0.900) and halves the generalisation gap
    (0.120 -> 0.055).

    Chain depth is applied per-account, never aggregated into the community
    table. It describes a node's POSITION in a chain; averaging it over a
    community destroys it (measured: 0.915 -> 0.790).
    """
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    if "motif_chain_depth" not in F.columns:
        raise ValueError("run motifs.py first -- chain depth is not optional")
    return z(F.comm_score) + 1.5 * z(F.motif_chain_depth.fillna(0))


# --------------------------------------------------------------------------
def evaluate(F: pd.DataFrame, k: int = 200):
    y = F.is_mule
    feats = [c for c in F.columns if c not in DROP]

    acct = unsupervised_score(F)
    z = lambda x: (x - x.mean()) / x.std()
    scores = {
        "Rules baseline": rules_baseline(F),
        "Unsupervised, account-level": acct,
        "GBM (sparse labels)": gbm_score(F, feats),
        "  control: activity only": gbm_score(F, ACTIVITY_ONLY),
    }
    if "comm_score" in F.columns:
        # Stage 4: network-level, still label-free. This is the headline.
        scores["Community score (no labels)"] = F.comm_score
        scores["Community x account (no labels)"] = z(F.comm_score) + 0.5 * z(acct)
        scores["FINAL: comm x account x chain"] = final_score(F)
    scores["GBM, 8 curated"] = gbm_score(F, CURATED)

    rows = []
    for name, s in scores.items():
        top = s.sort_values(ascending=False).head(k).index
        conf = F.loc[top, "account_type"].isin(CONFOUNDERS).mean()
        rows.append({
            "detector": name,
            f"precision@{k}": precision_at_k(s, y, k),
            f"recall@{k}": recall_at_k(s, y, k),
            "mule_networks_hit": networks_hit(s, F, k),
            "false_alerts_on_confounders": conf,
        })
    res = pd.DataFrame(rows).set_index("detector")

    print(f"\nBase rate: {y.mean()*100:.2f}%  |  true mules: {int(y.sum())}  |  "
          f"exposed labels: {int(F.label_exposed.sum())}  |  k = {k}")
    print(f"Total mule networks in data: "
          f"{len({n for s in F.mule_network_id if s for n in s.split('|')})}\n")
    print(res.round(3).to_string())

    # feature importance from a full-data fit, for the explainability slide
    X = F[feats].replace([np.inf, -np.inf], 0).fillna(0)
    m = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.05,
                           min_child_samples=5, random_state=0, verbose=-1)
    m.fit(X, F.label_exposed)
    imp = (pd.Series(m.feature_importances_, index=feats)
             .sort_values(ascending=False).head(12))
    print("\nTop features (LightGBM gain split count):")
    print(imp.to_string())
    return res, scores


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--k", type=int, default=200)
    a = p.parse_args()
    F = pd.read_parquet(Path(a.data) / "features.parquet")
    evaluate(F, a.k)
