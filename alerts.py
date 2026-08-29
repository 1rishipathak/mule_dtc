"""
Cross-bank alert payload.

The policy argument in the pitch is that no single bank sees a full layering
chain. The obvious objection is that banks cannot simply pool customer data.
So the payload is designed to carry the LEAST information that still lets a
receiving institution act:

  - the account handle is hashed, salted per reporting window
  - no name, no balance, no counterparty list, no transaction detail
  - evidence is a list of CODES, not raw feature values, so a receiving bank
    learns "this account sits deep in a fast-forwarding chain" and not
    "this account received Rs 4,80,000 at 14:32 from handle X"

That is the honest version of the cross-institution claim: we are not
proposing a shared transaction lake, we are proposing that derived risk
signals cross the boundary while customer data does not.

Every alert carries the evidence that produced it. An unexplained alert that
throttles someone's account is not something a bank can act on, and RBI's
FREE-AI principles put Explainability and Accountability front and centre.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SALT = "window-2026-02-14"     # rotated per reporting window

# feature -> (evidence code, human sentence, direction that is suspicious)
EVIDENCE = [
    ("motif_chain_depth", "CHAIN_DEPTH",
     "sits {v:.0f} hops deep in a time-respecting forwarding chain", "high"),
    ("p10_dwell", "FAST_EMPTY",
     "typically empties within {v:.0f} minutes of a credit", "low"),
    ("fast_flowthru_n", "FLOW_THROUGH",
     "{v:.0f} credits forwarded onward near-whole within minutes", "high"),
    ("reciprocity", "NO_RECIPROCITY",
     "only {v:.0%} of counterparties ever send money back", "low"),
    ("device_shared_by", "DEVICE_REUSE",
     "shares a device with {v:.0f} other accounts", "high"),
    ("age_min_at_first_txn", "NEW_ACCOUNT",
     "first activity {v:.0f} days after opening", "low"),
    ("stranger_in_share", "STRANGER_FUNDED",
     "{v:.0%} of inbound value comes from first-time senders", "high"),
]


def pseudonymise(handle: str) -> str:
    return "acct_" + hashlib.sha256((SALT + handle).encode()).hexdigest()[:16]


def evidence_for(row: pd.Series, ref: pd.DataFrame) -> list[dict]:
    """Only cite a signal when this account is genuinely extreme on it."""
    out = []
    for col, code, tmpl, direction in EVIDENCE:
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v):
            continue
        pct = (ref[col] < v).mean()
        extreme = pct > 0.97 if direction == "high" else pct < 0.03
        if not extreme:
            continue
        shown = v / 1440.0 if col == "age_min_at_first_txn" else v
        out.append({"code": code, "detail": tmpl.format(v=shown),
                    "percentile": round(float(pct), 3)})
    return out


def build_alerts(F: pd.DataFrame, tier_col="tier", limit=None) -> list[dict]:
    act = F[F[tier_col].isin(["freeze", "hold", "throttle"])]
    act = act.sort_values("score", ascending=False)
    if limit:
        act = act.head(limit)

    alerts = []
    for handle, row in act.iterrows():
        ev = evidence_for(row, F)
        if not ev:
            # No citable evidence means no defensible alert. Drop it rather
            # than ask another bank to act on an unexplained score.
            continue
        alerts.append({
            "schema": "mule-network-alert/0.1",
            "account_ref": pseudonymise(str(handle)),
            "risk_tier": row[tier_col],
            "score_percentile": round(float((F.score < row.score).mean()), 4),
            "network_ref": pseudonymise("net:" + str(row.get("lg_community", ""))),
            "network_size": int(F[F.lg_community == row.get("lg_community")].shape[0])
                            if "lg_community" in F.columns else None,
            "evidence": ev,
            "recommended_action": {
                "freeze": "Hold outbound; require step-up verification; file STR",
                "hold": "Hold outbound pending verification",
                "throttle": "Cap per-transaction and daily outflow limits",
            }[row[tier_col]],
            "basis": "network-level behavioural inference on derived signals; "
                     "no confirmed fraud report attached",
            "contains_pii": False,
        })
    return alerts


def main(data="data_big", limit=200):
    d = Path(data)
    F = pd.read_parquet(d / "tiered.parquet") if (d / "tiered.parquet").exists() else None
    if F is None:
        raise SystemExit("run response.py first to produce tiered.parquet")

    alerts = build_alerts(F, limit=limit)
    out = d / "alerts.json"
    out.write_text(json.dumps(alerts, indent=2))

    print(f"alerts generated : {len(alerts)}")
    by = pd.Series([a["risk_tier"] for a in alerts]).value_counts()
    print(by.to_string())
    codes = pd.Series([e["code"] for a in alerts for e in a["evidence"]]).value_counts()
    print("\nevidence codes cited:")
    print(codes.to_string())
    if alerts:
        print("\nexample payload:")
        print(json.dumps(alerts[0], indent=2))
    return alerts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data_big")
    p.add_argument("--limit", type=int, default=200)
    a = p.parse_args()
    main(a.data, a.limit)
