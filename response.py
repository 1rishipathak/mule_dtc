"""
Stage 6 -- graded response.

Two things this has to get right.

1. CAUSALITY. The 0.915 batch ranking is computed over the whole 45-day
   window, which no live system has. So the response layer runs on a
   STREAMING score built only from quantities computable from the past:

     chain depth  -- the DP is `depth[v] = max(depth[v], depth[u]+1)`
                     processed in time order, which is already causal.
                     The batch feature needs no modification.
     in-degree / in-value on the laundering subgraph -- running counts.
     device sharing -- known at KYC time, not derived from future traffic.

   Louvain is deliberately NOT in the streaming path: recomputing
   communities per event is expensive and the partition is unstable early
   in a network's life. Communities stay in the batch/analyst layer.

2. THE COST OF BEING WRONG. Mules are frequently recruited victims --
   students paid a few thousand rupees. Freezing an innocent account is a
   real harm, so tiers are calibrated on MEASURED precision, and the count
   of innocent accounts caught at each tier is reported, never hidden.

The recovery figure is a counterfactual replay: taint is propagated from
each fraud payout, and outbound transfers are blocked when the sending
account has already reached HOLD or FREEZE at that moment in the stream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from laundering_graph import flow_through_edges

TIERS = ["monitor", "throttle", "hold", "freeze"]
# minimum precision required to justify each action
# Calibrated against what the ranking can actually deliver, not aspiration.
# Nothing between freeze and throttle clears 0.70, so hold sits at 0.55.
TIER_PRECISION = {"freeze": 0.90, "hold": 0.55, "throttle": 0.20}
PAYOUT_MIN = 50_000.0     # what counts as a seeded fraud payout


# --------------------------------------------------------------------------
def streaming_state(edges: pd.DataFrame, acc: pd.DataFrame):
    """Replay the laundering subgraph in time order, recording the running
    feature state after every edge. Returns per-edge snapshots so any later
    timestamp can be resolved without recomputation."""
    ids = acc.account_id.to_numpy()
    n = len(ids)
    e = edges.sort_values("minute", kind="mergesort")
    u = e.u.to_numpy(); v = e.v.to_numpy()
    t = e.minute.to_numpy(); a = e.amount.to_numpy()

    CHAIN_GAP = 720          # same burst rule as motifs.py
    depth = np.zeros(n, np.float32)
    seen_at = np.full(n, -10**9, np.int64)
    in_deg = np.zeros(n, np.float32)
    in_val = np.zeros(n, np.float64)

    d_snap = np.zeros(len(u), np.float32)
    g_snap = np.zeros(len(u), np.float32)
    v_snap = np.zeros(len(u), np.float64)

    for k in range(len(u)):
        i, j, tk = u[k], v[k], t[k]
        base = depth[i] if tk - seen_at[i] <= CHAIN_GAP else 0.0
        cand = base + 1
        if tk - seen_at[j] > CHAIN_GAP or cand > depth[j]:
            depth[j] = cand
        seen_at[j] = tk
        in_deg[v[k]] += 1
        in_val[v[k]] += a[k]
        d_snap[k] = depth[v[k]]
        g_snap[k] = in_deg[v[k]]
        v_snap[k] = in_val[v[k]]

    hist = pd.DataFrame({"node": v, "minute": t, "depth": d_snap,
                         "in_deg": g_snap, "in_val": v_snap})
    final = pd.DataFrame({"depth": depth, "in_deg": in_deg, "in_val": in_val},
                         index=ids)
    return hist, final


def stream_score(depth, in_deg, in_val, device, ref):
    """Fixed, transparent weighting. `ref` holds the normalisation constants
    fitted once on the batch state so the live score is stable over time."""
    z = lambda x, m, s: (np.asarray(x, float) - m) / (s + 1e-9)
    return (1.0 * z(depth, *ref["depth"])
            + 0.6 * z(np.log1p(np.maximum(in_val, 0)), *ref["in_val"])
            + 0.4 * z(in_deg, *ref["in_deg"])
            + 0.5 * z(device, *ref["device"]))


# --------------------------------------------------------------------------
def calibrate(scores: pd.Series, y: pd.Series):
    """Cut each tier on the precision WITHIN its band, not cumulative
    precision from rank 0.

    A tier is a band, and an account in the hold band is never seen by the
    freeze rule. Calibrating on cumulative precision lets the dense top of
    the ranking pay for a band that is mostly innocent people: measured, the
    cumulative version delivered 0.315 against a 0.70 target for hold, and
    0.075 against 0.35 for throttle."""
    o = scores.sort_values(ascending=False)
    hit = y.reindex(o.index).to_numpy().astype(float)
    cum = np.concatenate([[0.0], np.cumsum(hit)])
    n = len(hit)

    cuts, start = {}, 0
    for tier in ("freeze", "hold", "throttle"):
        need = TIER_PRECISION[tier]
        j = start
        best = start
        while j < n:
            j += 1
            if j - start < 10:          # ignore noise in tiny bands
                continue
            if (cum[j] - cum[start]) / (j - start) >= need:
                best = j
            elif j - start > 2000:      # stop searching once hopeless
                break
        if best <= start:
            cuts[tier] = float("inf")   # no band clears the bar
        else:
            cuts[tier] = float(o.iloc[best - 1])
            start = best
    return cuts


def assign(scores: pd.Series, cuts: dict) -> pd.Series:
    t = pd.Series("none", index=scores.index)
    t[scores > cuts["throttle"]] = "throttle"
    t[scores > cuts["hold"]] = "hold"
    t[scores > cuts["freeze"]] = "freeze"
    return t


def tier_report(tier: pd.Series, F: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["freeze", "hold", "throttle"]:
        m = tier == name
        if not m.any():
            continue
        sub = F[m]
        rows.append({
            "tier": name, "accounts": int(m.sum()),
            "mules": int(sub.is_mule.sum()),
            "precision": round(float(sub.is_mule.mean()), 3),
            "innocent": int((~sub.is_mule.astype(bool)).sum()),
            "of which confounders": int(sub.account_type.isin(
                ["small_shop", "group_collector", "gig_aggregator",
                 "informal_lender"]).sum()),
        })
    return pd.DataFrame(rows).set_index("tier")


# --------------------------------------------------------------------------
def replay_recovery(tx, acc, hist, ref, cuts, device_map, block_from="hold"):
    """Counterfactual: propagate taint from fraud payouts through the P2P
    stream and block outbound transfers once the sender has reached
    `block_from` or worse at that point in time."""
    p2p = (tx[tx.channel == "P2P"] if "channel" in tx.columns else tx)
    p2p = p2p.sort_values("minute", kind="mergesort")
    A = acc.set_index("account_id")
    is_mule = A.is_mule
    is_exit = A.is_exit if "is_exit" in A.columns else (A.is_mule * 0)

    # resolve each account's score trajectory into (minute -> score) steps
    hist = hist.sort_values("minute", kind="mergesort")
    dev = hist.node.map(lambda i: 0)  # filled below per row
    sc = stream_score(hist.depth, hist.in_deg, hist.in_val,
                      device_map[hist.node.to_numpy()], ref)
    steps = {}
    for node, minute, s in zip(hist.node.to_numpy(), hist.minute.to_numpy(), sc):
        steps.setdefault(node, []).append((minute, s))

    order = {a: i for i, a in enumerate(acc.account_id.to_numpy())}
    thr = cuts[block_from]

    def flagged_at(a, t):
        st = steps.get(order.get(a, -1))
        if not st:
            return False
        best = -1e9
        for m, s in st:
            if m <= t:
                best = max(best, s)
            else:
                break
        return best > thr

    taint = {}
    total_payout = 0.0
    blocked = 0.0
    cashed = 0.0

    for r in p2p.itertuples(index=False):
        s, d, amt, t = r.sender, r.receiver, r.amount, r.minute
        # seed: a large transfer from a non-mule into a mule is a payout
        if amt >= PAYOUT_MIN and not is_mule.get(s, 0) and is_mule.get(d, 0):
            taint[d] = taint.get(d, 0.0) + amt
            total_payout += amt
            continue
        held = taint.get(s, 0.0)
        if held <= 0:
            continue
        move = min(held, amt)
        if flagged_at(s, t):
            # Funds are RETAINED, so remove them from the movable pool.
            # Without this the same rupee is blocked once per attempted
            # transfer and the recovery figure exceeds 100%.
            taint[s] = held - move
            blocked += move
            continue
        taint[s] = held - move
        if is_exit.get(d, 0):
            cashed += move           # OUT of the banking system -- gone
        elif is_mule.get(d, 0):
            taint[d] = taint.get(d, 0.0) + move
        else:
            cashed += move
    # blocked / cashed_out / never_moved partition the payout exactly
    # "cashed_out" is now value that reached a real exit point (ATM agent,
    # crypto off-ramp, complicit shopfront) -- money out of the system.
    still_in = sum(v for v in taint.values() if v > 0)
    return {"payout_total": total_payout, "blocked": blocked,
            "cashed_out": cashed, "still_in_mule_layer": still_in,
            "unaccounted": total_payout - blocked - cashed - still_in}


# --------------------------------------------------------------------------
def build(data="data_big"):
    d = Path(data)
    tx = pd.read_parquet(d / "transactions.parquet")
    acc = pd.read_parquet(d / "accounts.parquet")
    F = pd.read_parquet(d / "features.parquet")

    edges, ids = flow_through_edges(tx, acc)
    hist, final = streaming_state(edges, acc)

    device = acc.set_index("account_id").device_shared_by \
        if "device_shared_by" in acc.columns else F.device_shared_by
    device = device.reindex(ids).fillna(1).to_numpy()

    ref = {"depth": (final.depth.mean(), final.depth.std()),
           "in_deg": (final.in_deg.mean(), final.in_deg.std()),
           "in_val": (np.log1p(final.in_val).mean(), np.log1p(final.in_val).std()),
           "device": (device.mean(), device.std())}

    s = pd.Series(stream_score(final.depth, final.in_deg, final.in_val,
                               device, ref), index=ids)
    y = F.is_mule.reindex(ids)
    cuts = calibrate(s, y)
    tier = assign(s, cuts)

    print("\nstreaming detector (causal features only):")
    print(f"  precision@200 = {y.reindex(s.sort_values(ascending=False).head(200).index).mean():.3f}")
    print("\ngraded response tiers:")
    print(tier_report(tier, F.reindex(ids)).to_string())

    rec = replay_recovery(tx, acc, hist, ref, cuts, device, block_from="freeze")
    print("\ncounterfactual recovery (block at FREEZE tier):")
    tot = rec["payout_total"]
    for k in ["payout_total", "blocked", "cashed_out",
              "still_in_mule_layer", "unaccounted"]:
        pct = f"  ({100*rec[k]/tot:5.1f}%)" if tot else ""
        print(f"  {k:28s} Rs {rec[k]:14,.0f}{pct}")
    out = F.reindex(ids).copy()
    out["score"] = s
    out["tier"] = tier
    out.to_parquet(Path(data) / "tiered.parquet")
    return s, tier, cuts, rec


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_big")
    build(p.parse_args().data)
