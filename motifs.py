"""
Stage 5 -- motif features.

The four typologies in the simulator each leave a distinct fingerprint in the
flow-through subgraph. These are computed on the LAUNDERING graph (4k edges),
not the full transfer graph, so everything here is cheap.

  fan-out star   -> one node pushing to many, inside one short window
  peel chain     -> long time-respecting path, each hop retaining a cut
  round-trip     -> closed walks through the node
  smurfing       -> many transfers parked just under a reporting threshold

Cycles are counted as closed walks via sparse matrix powers rather than
enumerating simple cycles, which does not terminate reliably on real graphs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from laundering_graph import flow_through_edges

STAR_WINDOW = 30       # minutes
CHAIN_GAP = 720        # minutes: a chain is one burst, not a 45-day history
SMURF_LO, SMURF_HI = 4000.0, 10000.0


def motif_features(edges: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    n = len(ids)
    u = edges.u.to_numpy()
    v = edges.v.to_numpy()
    t = edges.minute.to_numpy()
    a = edges.amount.to_numpy()

    F = pd.DataFrame(index=ids)
    F.index.name = "account_id"

    # ---- degrees in the laundering graph -------------------------------
    F["lg_out_deg"] = np.bincount(u, minlength=n)
    F["lg_in_deg"] = np.bincount(v, minlength=n)
    F["lg_out_uniq"] = pd.Series(v).groupby(u).nunique().reindex(range(n)).fillna(0).to_numpy()
    F["lg_in_uniq"] = pd.Series(u).groupby(v).nunique().reindex(range(n)).fillna(0).to_numpy()

    # ---- fan-out star: most distinct targets inside one short window ----
    star = np.zeros(n)
    ordr = np.lexsort((t, u))
    us, ts, vs = u[ordr], t[ordr], v[ordr]
    b = np.searchsorted(us, np.arange(n + 1))
    for i in range(n):
        lo, hi = b[i], b[i + 1]
        if hi - lo < 2:
            continue
        tt, vv = ts[lo:hi], vs[lo:hi]
        j = np.searchsorted(tt, tt + STAR_WINDOW, "right")
        star[i] = max(len(set(vv[k:j[k]])) for k in range(len(tt)))
    F["motif_star_out"] = star

    # ---- peel chain: longest time-respecting path ending at the node ----
    # Depth is BURST-SCOPED. Without the gap rule the DP never resets, so
    # separate laundering runs weeks apart concatenate into one apparent
    # path and the feature reports depths of 19+ against a configured
    # layering depth of 2-4. The score still worked, but the explanation
    # shipped to another bank was false.
    e_ord = np.argsort(t, kind="stable")
    depth = np.zeros(n)
    seen_at = np.full(n, -10**9, dtype=np.int64)   # when depth was last set
    best = np.zeros(n)
    for k in e_ord:
        i, j, tk = u[k], v[k], t[k]
        base = depth[i] if tk - seen_at[i] <= CHAIN_GAP else 0.0
        cand = base + 1
        if tk - seen_at[j] > CHAIN_GAP or cand > depth[j]:
            depth[j] = cand
        seen_at[j] = tk
        if depth[j] > best[j]:
            best[j] = depth[j]
    F["motif_chain_depth"] = best

    # ---- round trips: closed walks of length 2, 3, 4 --------------------
    A = sp.csr_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    A.data[:] = 1.0
    A2 = A @ A
    F["motif_cycle2"] = A2.diagonal()
    A3 = A2 @ A
    F["motif_cycle3"] = A3.diagonal()
    F["motif_cycle4"] = (A3 @ A).diagonal()

    # ---- smurfing: value parked just under a reporting threshold --------
    smurf = ((a >= SMURF_LO) & (a <= SMURF_HI)).astype(float)
    F["motif_smurf_in"] = np.bincount(v, weights=smurf, minlength=n)
    F["motif_smurf_out"] = np.bincount(u, weights=smurf, minlength=n)
    tot_in = np.bincount(v, minlength=n)
    F["motif_smurf_share"] = F.motif_smurf_in / np.maximum(tot_in, 1)

    # ---- value actually moved through the laundering graph --------------
    F["lg_val_in"] = np.bincount(v, weights=a, minlength=n)
    F["lg_val_out"] = np.bincount(u, weights=a, minlength=n)
    return F


def build(data="data_big"):
    d = Path(data)
    tx = pd.read_parquet(d / "transactions.parquet")
    acc = pd.read_parquet(d / "accounts.parquet")
    F = pd.read_parquet(d / "features.parquet")

    edges, ids = flow_through_edges(tx, acc)
    M = motif_features(edges, ids)
    F = F.drop(columns=[c for c in M.columns if c in F.columns]).join(M)
    F.to_parquet(d / "features.parquet")
    print(f"  motif features added: {list(M.columns)}")
    return F


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_big")
    F = build(p.parse_args().data)
    from sklearn.metrics import roc_auc_score
    y = F.is_mule.to_numpy()
    print("\nunivariate AUC of each motif:")
    for c in F.columns:
        if c.startswith(("motif_", "lg_")):
            s = roc_auc_score(y, F[c].fillna(0).to_numpy())
            print(f"  {c:22s} {max(s, 1-s):.3f}")
