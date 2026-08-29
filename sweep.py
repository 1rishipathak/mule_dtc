"""
Sensitivity harness.

Runs the whole pipeline for a given (scale, seed) and emits one row of
metrics. Two questions it exists to answer:

  1. Does precision hold as the base rate falls toward the real ~0.1%?
     A detector that only works at an inflated base rate is not a detector.
  2. How much does the headline number move across random seeds?
     One seed is an anecdote.

Everything is re-derived per run. Nothing is cached across configurations,
so a run cannot borrow tuning from another.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

import features as feat
import laundering_graph as lg
import motifs as mot
from model import (final_score, precision_at_k, recall_at_k, networks_hit,
                   rules_baseline, gbm_score, CURATED, CONFOUNDERS)
from simulate import Config, Simulator


def one_run(scale: float, seed: int, k: int = 200, keep=False) -> dict:
    out = Path(f"/tmp/sweep_s{scale}_{seed}")
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config(seed=seed, scale=scale)
    tx, acc = Simulator(cfg).run()
    tx.to_parquet(out / "transactions.parquet", index=False)
    acc.to_parquet(out / "accounts.parquet", index=False)

    seeds = acc.loc[acc.label_exposed == 1, "account_id"].tolist()
    b = feat.behavioural_features(tx, acc)
    r = feat.relational_features(tx, acc)
    g, _ = feat.graph_features(tx, acc, seeds)
    F = acc.set_index("account_id")[["account_type", "is_mule", "label_exposed",
                                     "mule_network_id"]].join([b, r, g]).fillna(0)

    edges, ids = lg.flow_through_edges(tx, acc)
    F["lg_community"] = lg.detect_communities(edges, ids).reindex(F.index).fillna(-1).astype(int)
    F["comm_score"], _ = lg.score_communities(F)
    M = mot.motif_features(edges, ids)
    F = F.drop(columns=[c for c in M.columns if c in F.columns]).join(M)

    y = F.is_mule
    s = final_score(F)
    top = s.sort_values(ascending=False).head(k).index
    base = float(y.mean())

    row = {
        "scale": scale, "seed": seed,
        "accounts": len(F), "mules": int(y.sum()),
        "base_rate_pct": round(base * 100, 3),
        "labels": int(F.label_exposed.sum()),
        "precision_at_k": round(precision_at_k(s, y, k), 3),
        "lift": round(precision_at_k(s, y, k) / base, 1) if base else None,
        "recall_at_k": round(recall_at_k(s, y, k), 3),
        "networks_hit": networks_hit(s, F, k),
        "conf_fp": round(float(F.loc[top, "account_type"].isin(CONFOUNDERS).mean()), 3),
        "rules_precision": round(precision_at_k(rules_baseline(F), y, k), 3),
        "gbm_precision": round(precision_at_k(gbm_score(F, CURATED), y, k), 3),
    }
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--append", default="/home/claude/mule_detect/sweep_results.jsonl")
    a = p.parse_args()

    row = one_run(a.scale, a.seed, a.k)
    print(json.dumps(row))
    with open(a.append, "a") as f:
        f.write(json.dumps(row) + "\n")
