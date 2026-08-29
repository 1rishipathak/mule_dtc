"""
Stage 4 -- score NETWORKS, not accounts.

Two ideas, in order:

1. The laundering graph is NOT the transfer graph.
   Running community detection on all P2P transfers returns "the mule crew
   plus everyone they know" -- a mule's ordinary friends get pulled in by
   cover traffic, capping precision around 0.35.

   So first build the FLOW-THROUGH SUBGRAPH: keep only edges u->v where v
   forwarded a comparable amount onward within a short window. That is the
   operational definition of a layering hop, and it is time-respecting by
   construction. Cover traffic drops out because ordinary people keep what
   they receive.

2. Then detect communities on that subgraph, score each COMMUNITY, and
   broadcast the score to its members. With 14 networks and ~30 exposed
   labels there is nowhere near enough signal to fit per-account; there is
   plenty to separate a few hundred candidate groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd

# Tuned by sweep. The value floor stays at 5k, NOT higher: the smurfing
# typology deliberately uses sub-threshold 4k-9.5k transfers, and a 20k
# floor gives a cleaner graph by silently deleting that whole typology.
FLOW_WINDOW = 15      # minutes: credit -> onward debit
FLOW_RATIO = 0.80     # fraction of the credit that must move on
FLOW_MIN_AMOUNT = 5000.0
MIN_COMM = 4          # a "network" smaller than this is not a network
MAX_COMM = 200        # networks merge via reused mules and shared cash-outs


# --------------------------------------------------------------------------
def flow_through_edges(tx: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    """Flag each credit that was forwarded onward fast, then return the
    resulting edge list. This is the laundering subgraph."""
    p2p = tx[tx.channel == "P2P"] if "channel" in tx.columns else tx
    ids = acc.account_id.to_numpy()
    cats = pd.CategoricalDtype(ids)
    s = p2p.sender.astype(cats).cat.codes.to_numpy(np.int32)
    r = p2p.receiver.astype(cats).cat.codes.to_numpy(np.int32)
    t = p2p.minute.to_numpy(np.int32)
    a = p2p.amount.to_numpy(np.float64)

    # debits, grouped by sender
    d_ord = np.lexsort((t, s))
    ds, dt, da = s[d_ord], t[d_ord], a[d_ord]
    d_bounds = np.searchsorted(ds, np.arange(len(ids) + 1))
    d_cum = np.concatenate([[0.0], np.cumsum(da)])

    is_flow = np.zeros(len(s), bool)
    order = np.argsort(r, kind="stable")
    r_sorted = r[order]                      # sort ONCE, not per account
    c_bounds = np.searchsorted(r_sorted, np.arange(len(ids) + 1))

    for i in range(len(ids)):
        lo_d, hi_d = d_bounds[i], d_bounds[i + 1]
        if hi_d <= lo_d:
            continue
        ci = order[c_bounds[i]:c_bounds[i + 1]]
        if not len(ci):
            continue
        ct, ca = t[ci], a[ci]
        win_lo = np.searchsorted(dt[lo_d:hi_d], ct, "left") + lo_d
        win_hi = np.searchsorted(dt[lo_d:hi_d], ct + FLOW_WINDOW, "right") + lo_d
        moved = d_cum[win_hi] - d_cum[win_lo]
        is_flow[ci] = (moved >= FLOW_RATIO * ca) & (ca >= FLOW_MIN_AMOUNT)

    # ---- terminal hop ---------------------------------------------------
    # The flow-through test asks whether the RECEIVER forwarded money onward,
    # so it structurally cannot see the last node in a chain -- which is
    # exactly the cash-out point where funds leave the banking system.
    # Measured: all 14 exit accounts had in-degree 0 and were absent from the
    # graph entirely. A chain has to include its own endpoint, so a transfer
    # OUT of an already-identified laundering node, above the value floor,
    # is kept even when the receiver never forwards.
    # A terminal hop must belong to the SAME burst as the laundering credit
    # that funded it -- an outbound transfer shortly AFTER a flow-through
    # credit into the sender. Without the timing constraint, any later
    # payment above the floor qualifies: measured, 42,292 spurious edges and
    # the graph grew from 3.8k to 26.5k nodes, erasing the enrichment.
    in_graph = np.zeros(len(ids), bool)
    in_graph[s[is_flow]] = True
    in_graph[r[is_flow]] = True

    fc_ord = np.argsort(np.where(is_flow, r, len(ids)), kind="stable")
    fc_ord = fc_ord[is_flow[fc_ord]]
    fc_node, fc_t = r[fc_ord], t[fc_ord]
    fc_b = np.searchsorted(fc_node, np.arange(len(ids) + 1))

    is_terminal = np.zeros(len(s), bool)
    cand = np.flatnonzero((~is_flow) & in_graph[s] & (a >= FLOW_MIN_AMOUNT)
                          & (~in_graph[r]))
    for k in cand:
        i = s[k]
        lo, hi = fc_b[i], fc_b[i + 1]
        if hi <= lo:
            continue
        j = np.searchsorted(fc_t[lo:hi], t[k], "right")
        if j > 0 and t[k] - fc_t[lo + j - 1] <= FLOW_WINDOW:
            is_terminal[k] = True
    keep = is_flow | is_terminal
    print(f"  terminal-hop edges: {int(is_terminal.sum()):,}")

    e = pd.DataFrame({"u": s[keep], "v": r[keep],
                      "amount": a[keep], "minute": t[keep]})
    print(f"  laundering edges: {len(e):,} of {len(s):,} P2P transfers "
          f"({100*len(e)/len(s):.1f}%)")
    return e, ids


# --------------------------------------------------------------------------
def detect_communities(edges: pd.DataFrame, ids: np.ndarray) -> pd.Series:
    agg = (edges.groupby(["u", "v"])
                .agg(w=("amount", "sum"), k=("amount", "size"))
                .reset_index())
    agg = agg[agg.u != agg.v]
    lo = np.minimum(agg.u.to_numpy(), agg.v.to_numpy())
    hi = np.maximum(agg.u.to_numpy(), agg.v.to_numpy())
    und = (pd.DataFrame({"a": lo, "b": hi, "w": agg.w.to_numpy()})
             .groupby(["a", "b"], as_index=False).w.sum())
    G = nx.from_pandas_edgelist(und, "a", "b", edge_attr="w")
    print(f"  laundering graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    part = community_louvain.best_partition(G, weight="w", random_state=0)
    s = pd.Series({ids[k]: v for k, v in part.items()}, name="lg_community")
    return s.reindex(ids).fillna(-1).astype(int)


# --------------------------------------------------------------------------
def score_communities(F: pd.DataFrame, comm_col: str = "lg_community"):
    """Aggregate member features per community, z-score, and combine with
    signs that encode the laundering hypothesis. Deliberately a transparent
    linear score -- a black box that freezes accounts is a non-starter."""
    valid = F[F[comm_col] >= 0]
    g = valid.groupby(comm_col)
    C = pd.DataFrame({
        "size": g.size(),
        "dwell": g.p10_dwell.median(),
        "recip": g.reciprocity.mean(),
        "device": g.device_shared_by.max(),
        "flowthru": g.fast_flowthru_n.mean(),
        "passthru": g.med_matched_passthru.mean(),
        "age": g.age_min_at_first_txn.median(),
        "peak_val": g.peak_in_val.sum(),
    })
    C = C[(C["size"] >= MIN_COMM) & (C["size"] <= MAX_COMM)]
    if C.empty:
        return pd.Series(0.0, index=F.index), C

    z = (C - C.mean()) / C.std().replace(0, 1)
    C["score"] = (-1.2 * z.dwell        # empties fast
                  - 1.0 * z.recip       # nobody pays anybody back
                  + 1.0 * z.device      # one handset, many accounts
                  + 1.0 * z.flowthru    # credits forwarded whole
                  + 0.6 * z.passthru
                  - 0.6 * z.age         # freshly opened
                  + 0.4 * z.peak_val)   # real money moved
    out = F[comm_col].map(C["score"]).fillna(C["score"].min() - 1)
    return out, C.sort_values("score", ascending=False)


# --------------------------------------------------------------------------
def build(data="data_big"):
    d = Path(data)
    tx = pd.read_parquet(d / "transactions.parquet")
    acc = pd.read_parquet(d / "accounts.parquet")
    F = pd.read_parquet(d / "features.parquet")

    edges, ids = flow_through_edges(tx, acc)
    comm = detect_communities(edges, ids)
    F["lg_community"] = comm.reindex(F.index).fillna(-1).astype(int)

    # how much of each laundering community is actually mule?
    v = F[F.lg_community >= 0]
    cs = v.groupby("lg_community").agg(n=("is_mule", "size"),
                                       mules=("is_mule", "sum"))
    cs = cs[(cs.n >= MIN_COMM) & (cs.n <= MAX_COMM)]
    hit = cs[cs.mules > 0]
    print(f"  candidate communities: {len(cs)}  |  containing mules: {len(hit)}")
    if len(hit):
        print(f"  mule purity of those: "
              f"{(hit.mules.sum() / hit.n.sum()):.2f}  "
              f"(mules captured: {int(hit.mules.sum())}/{int(F.is_mule.sum())})")

    score, C = score_communities(F)
    F["comm_score"] = score
    F.to_parquet(d / "features.parquet")
    return F, C


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_big")
    F, C = build(p.parse_args().data)
    print("\ntop-scoring communities:")
    print(C.head(8).round(2).to_string())
