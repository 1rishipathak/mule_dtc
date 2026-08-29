"""
Feature engineering for network-level mule detection.

Design note (this is the crux of the project):
    Mule accounts belong to real people and carry real cover traffic. Any
    feature aggregated over an account's full history is therefore DILUTED
    by that cover traffic and does not separate mules from ordinary users.
    Every behavioural feature here is computed over a short WINDOW and
    reduced with an extreme statistic (min / max / peak), not a mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import network as nx
import scipy.sparse as sp
import numpy as np
import pandas as pd
import community as community_louvain

WINDOW_MIN = 60          # burst window, minutes
PASSTHRU_W = 180         # matched pass-through horizon, minutes
GRAPH_MIN_VALUE = 1000.0 # prune trivial P2P chatter from the laundering graph


# --------------------------------------------------------------------------
# 1. Windowed behavioural features
# --------------------------------------------------------------------------
def behavioural_features(tx: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    """Integer-coded, numpy-only. Strings are the memory killer at scale."""
    ids = acc.account_id.to_numpy()
    n = len(ids)
    cats = pd.CategoricalDtype(ids)
    s_code = tx.sender.astype(cats).cat.codes.to_numpy(np.int32)
    r_code = tx.receiver.astype(cats).cat.codes.to_numpy(np.int32)
    minute = tx.minute.to_numpy(np.int32)
    amount = tx.amount.to_numpy(np.float32)

    acct = np.concatenate([r_code, s_code])
    mins = np.concatenate([minute, minute])
    amts = np.concatenate([amount, amount])
    sign = np.concatenate([np.ones(len(r_code), np.int8),
                           -np.ones(len(s_code), np.int8)])
    del s_code, r_code

    order = np.lexsort((mins, acct))
    acct, mins, amts, sign = acct[order], mins[order], amts[order], sign[order]
    del order
    bounds = np.searchsorted(acct, np.arange(n + 1))

    keys = ["min_dwell", "p10_dwell", "fast_flowthru_n", "med_matched_passthru",
            "peak_in_val", "peak_in_n", "peak_out_n", "peak_window_passthru",
            "burstiness", "night_share", "n_txn", "tot_in", "tot_out"]
    out = {k: np.zeros(n, np.float32) for k in keys}

    for i in range(n):
        lo_i, hi_i = bounds[i], bounds[i + 1]
        if hi_i <= lo_i:
            out["min_dwell"][i] = out["p10_dwell"][i] = 1e5
            continue
        m, sg, am = mins[lo_i:hi_i], sign[lo_i:hi_i], amts[lo_i:hi_i]
        out["n_txn"][i] = len(m)
        out["tot_in"][i] = am[sg > 0].sum()
        out["tot_out"][i] = am[sg < 0].sum()

        ci = np.flatnonzero(sg > 0)
        di = np.flatnonzero(sg < 0)
        if len(ci) and len(di):
            dm = m[di]
            j = np.searchsorted(dm, m[ci], side="left")
            ok = j < len(dm)
            if ok.any():
                dwell = dm[j[ok]] - m[ci][ok]
                out["min_dwell"][i] = dwell.min()
                out["p10_dwell"][i] = np.percentile(dwell, 10)
            else:
                out["min_dwell"][i] = out["p10_dwell"][i] = 1e5
            dcum = np.concatenate([[0.0], np.cumsum(am[di], dtype=np.float64)])
            lo = np.searchsorted(dm, m[ci], side="left")
            hi = np.searchsorted(dm, m[ci] + PASSTHRU_W, side="right")
            ratio = np.clip((dcum[hi] - dcum[lo]) /
                            np.maximum(am[ci], 1.0), 0, 1.5)
            out["med_matched_passthru"][i] = np.median(ratio)
            nxt = np.where(lo < len(dm), dm[np.minimum(lo, len(dm) - 1)] - m[ci], 1e5)
            out["fast_flowthru_n"][i] = ((ratio > 0.75) & (nxt < 30)).sum()
        else:
            out["min_dwell"][i] = out["p10_dwell"][i] = 1e5

        if len(m) > 1:
            lo = np.searchsorted(m, m - WINDOW_MIN, side="left")
            hi = np.searchsorted(m, m + WINDOW_MIN, side="right")
            b = int(np.argmax(hi - lo))
            ws, wa = sg[lo[b]:hi[b]], am[lo[b]:hi[b]]
            wi, wo = wa[ws > 0].sum(), wa[ws < 0].sum()
            out["peak_in_val"][i] = wi
            out["peak_in_n"][i] = (ws > 0).sum()
            out["peak_out_n"][i] = (ws < 0).sum()
            out["peak_window_passthru"][i] = wo / wi if wi > 0 else 0.0
            gaps = np.diff(m)
            out["burstiness"][i] = gaps.std() / (gaps.mean() + 1e-9)
        out["night_share"][i] = np.mean((m % 1440 < 360) | (m % 1440 > 1320))

    f = pd.DataFrame(out, index=ids)
    f.index.name = "account_id"
    return f


# --------------------------------------------------------------------------
# 2. Counterparty / relational features
# --------------------------------------------------------------------------
def relational_features(tx: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    fan_in = tx.groupby("receiver").sender.nunique().rename("fan_in")
    fan_out = tx.groupby("sender").receiver.nunique().rename("fan_out")
    in_n = tx.groupby("receiver").size().rename("in_n")
    out_n = tx.groupby("sender").size().rename("out_n")

    # reciprocity via sparse adjacency: fraction of counterparties with
    # flow in BOTH directions. Real peers settle up; mules almost never do.
    ids = acc.account_id.to_numpy()
    cats = pd.CategoricalDtype(ids)
    si = tx.sender.astype(cats).cat.codes.to_numpy(np.int32)
    ri = tx.receiver.astype(cats).cat.codes.to_numpy(np.int32)
    n = len(ids)
    A = sp.csr_matrix((np.ones(len(si), np.int8), (si, ri)), shape=(n, n))
    A.data[:] = 1
    A.sum_duplicates()
    A.data[:] = 1
    mutual = A.multiply(A.T)                       # edges present both ways
    undirected = ((A + A.T) > 0).astype(np.int8)
    deg = np.asarray(undirected.sum(axis=1)).ravel()
    mut = np.asarray(((mutual + mutual.T) > 0).astype(np.int8).sum(axis=1)).ravel()
    recip = pd.Series(np.divide(mut, np.maximum(deg, 1)), index=ids,
                      name="reciprocity")
    del A, mutual, undirected

    # share of inbound VALUE arriving from a first-time sender
    t = tx.sort_values("minute", kind="mergesort")
    first = ~t.duplicated(["receiver", "sender"], keep="first")
    newval = t[first].groupby("receiver").amount.sum()
    allval = t.groupby("receiver").amount.sum()
    stranger = (newval / allval).rename("stranger_in_share")

    # device sharing
    dev_ct = acc.groupby("device_id").account_id.count().rename("dev_n")
    dev = acc.set_index("account_id").device_id.map(dev_ct).rename("device_shared_by")

    # account age at first activity
    first_act = tx.groupby("receiver").minute.min().rename("first_min")
    a = acc.set_index("account_id")
    age = (-a.open_day * 1440.0)
    age_at_first = (age.add(first_act, fill_value=0)).rename("age_min_at_first_txn")

    df = pd.concat([fan_in, fan_out, in_n, out_n, recip, stranger, dev,
                    age_at_first], axis=1)
    df.index.name = "account_id"
    return df.reindex(acc.account_id).fillna(0)


# --------------------------------------------------------------------------
# 3. Graph-topological features
# --------------------------------------------------------------------------
def graph_features(tx: pd.DataFrame, acc: pd.DataFrame,
                   seeds: list[str] | None = None) -> pd.DataFrame:
    # Build the laundering graph on P2P rails only. UPI flags P2M separately,
    # and merchant payments are not laundering channels. Kirana shops running
    # on a PERSONAL VPA remain in the graph -- they are the hard negatives.
    p2p = tx[tx.channel == "P2P"] if "channel" in tx.columns else tx
    agg = (p2p.groupby(["sender", "receiver"])
             .agg(w=("amount", "sum"), k=("amount", "size"),
                  t_first=("minute", "min"), t_last=("minute", "max"))
             .reset_index())
    agg = agg[agg.w >= GRAPH_MIN_VALUE]   # drop trivial chatter
    print(f"  graph: {len(agg):,} edges after P2P + value pruning")

    G = nx.DiGraph()
    G.add_nodes_from(acc.account_id)
    for r in agg.itertuples(index=False):
        G.add_edge(r.sender, r.receiver, w=r.w, k=r.k,
                   t_first=r.t_first, t_last=r.t_last)
    U = G.to_undirected()

    # --- Louvain communities -------------------------------------------
    part = community_louvain.best_partition(U, weight="w", random_state=0)
    comm = pd.Series(part, name="community")

    # community quality: dense + non-reciprocal + fast = suspicious
    tmp = acc.set_index("account_id").join(comm)
    csize = tmp.groupby("community").size().rename("comm_size")

    pairs_f = set(map(tuple, agg[["sender", "receiver"]].to_numpy()))
    edge_recip = {}
    for u, v in G.edges():
        cu = part.get(u)
        if cu is not None and cu == part.get(v):
            edge_recip.setdefault(cu, []).append(1 if (v, u) in pairs_f else 0)
    comm_recip = pd.Series({c: float(np.mean(x)) for c, x in edge_recip.items()},
                           name="comm_reciprocity")

    # --- core / centrality ---------------------------------------------
    core = pd.Series(nx.core_number(nx.Graph(U)), name="k_core")
    btw = pd.Series(nx.betweenness_centrality(U, k=min(150, U.number_of_nodes()),
                                              seed=0, weight=None),
                    name="betweenness")

    # --- time-respecting distance to a seeded fraud account -------------
    trd = pd.Series(np.nan, index=list(G.nodes()), name="time_resp_dist_to_seed")
    if seeds:
        for s in seeds:
            if s not in G:
                continue
            # forward BFS obeying non-decreasing timestamps
            frontier = [(s, -1)]
            dist = {s: 0}
            while frontier:
                nxt = []
                for u, tu in frontier:
                    for v in G.successors(u):
                        te = G[u][v]["t_first"]
                        if te >= tu and v not in dist:
                            dist[v] = dist[u] + 1
                            nxt.append((v, te))
                frontier = nxt
            for k, v in dist.items():
                trd[k] = min(v, trd[k]) if pd.notna(trd[k]) else v
    trd = trd.fillna(99)

    df = pd.concat([comm, core, btw, trd], axis=1)
    df["comm_size"] = df.community.map(csize).fillna(0)
    df["comm_reciprocity"] = df.community.map(comm_recip).fillna(0)
    df.index.name = "account_id"
    return df.reindex(acc.account_id).fillna(0), G


def build(data="data"):
    d = Path(data)
    tx = pd.read_parquet(d / "transactions.parquet")
    acc = pd.read_parquet(d / "accounts.parquet")

    # seeds = the delayed, exposed fraud reports only
    seeds = acc.loc[acc.label_exposed == 1, "account_id"].tolist()

    b = behavioural_features(tx, acc)
    r = relational_features(tx, acc)
    g, G = graph_features(tx, acc, seeds)

    F = acc.set_index("account_id")[["account_type", "is_mule", "label_exposed",
                                     "mule_network_id"]].join([b, r, g])
    F["passthru_agg"] = F.tot_out / F.tot_in.replace(0, np.nan)
    F = F.fillna(0)
    F.to_parquet(d / "features.parquet")
    print(f"features: {F.shape[0]:,} accounts x "
          f"{F.select_dtypes('number').shape[1]} numeric features")
    return F


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    build(p.parse_args().data)
