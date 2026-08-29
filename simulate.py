"""
UPI transaction simulator for network-level mule detection.

Generates a temporal directed transfer graph containing:
  - a legitimate population (salaried / gig / student / merchant)
  - a CONFOUNDER population: legitimate accounts whose surface behaviour
    mimics mules (high fan-in shops, group collectors, shared devices).
    These are the hard negatives that make Precision@k meaningful.
  - injected mule networks using four published laundering typologies.

Ground truth is known by construction, but only a small, delayed subset
of labels is exposed to supervised training, to simulate real label scarcity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PER_DAY = 24 * 60


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = 42
    days: int = 45
    scale: float = 1.0   # multiplies the LEGIT population only -> controls base rate

    # population sizes
    n_salaried: int = 3200
    n_gig: int = 1400
    n_student: int = 1100
    n_merchant: int = 400
    n_employer: int = 60

    # confounders (legitimate but mule-shaped)
    n_small_shop: int = 140          # very high fan-in, low reciprocity
    n_group_collector: int = 110     # pass-through ~1.0, short dwell
    n_gig_aggregator: int = 60       # high velocity, many counterparties
    n_informal_lender: int = 50      # high pass-through, many uniques

    # social structure
    cluster_size: int = 30           # legit accounts form real communities
    peers_min: int = 3
    peers_max: int = 8
    p_peer_outside_cluster: float = 0.15

    # legit device sharing (families) -- keeps device reuse from being a
    # perfect giveaway
    p_shared_household_device: float = 0.09

    # mule networks
    n_mule_networks: int = 14
    mules_per_network: tuple[int, int] = (8, 34)
    layering_depth: tuple[int, int] = (2, 4)
    dwell_seconds: tuple[int, int] = (25, 1500)      # 25s - 25min
    cut_pct: tuple[float, float] = (0.02, 0.12)      # retained per hop
    p_network_shares_device: float = 0.6
    devices_per_network: tuple[int, int] = (1, 4)
    mule_reuse_rate: float = 0.18                    # mules serving 2+ networks
    p_mule_is_aged_dormant: float = 0.3
    # cash-out: where the money actually LEAVES the banking system.
    # ATM agents, crypto P2P traders, complicit shopfronts.
    n_exit_points: int = 14
    exits_per_network: tuple[int, int] = (1, 3)
    exit_share: tuple[float, float] = (0.80, 0.95)
    payout_amount: tuple[int, int] = (50_000, 600_000)
    payouts_per_network: tuple[int, int] = (6, 20)

    # label exposure (the scarcity simulation)
    label_exposure_rate: float = 0.15
    report_delay_days: tuple[int, int] = (3, 11)

    typology_mix: dict = field(default_factory=lambda: {
        "fan_out_star": 0.40,
        "peel_chain": 0.25,
        "round_trip_cycle": 0.15,
        "smurfing": 0.20,
    })


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------
class Simulator:
    def __init__(self, cfg: Config):
        if cfg.scale != 1.0:
            for f in ("n_salaried", "n_gig", "n_student", "n_merchant",
                      "n_small_shop", "n_group_collector", "n_gig_aggregator",
                      "n_informal_lender"):
                setattr(cfg, f, max(1, int(getattr(cfg, f) * cfg.scale)))
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.T = cfg.days * MIN_PER_DAY
        self.accounts: list[dict] = []
        self._idx: dict[str, dict] = {}
        self.txns: list[tuple] = []
        self._next_id = 0
        self._next_device = 0

    # ---- account creation -------------------------------------------------
    def _new_account(self, kind: str, **kw) -> str:
        aid = f"A{self._next_id:06d}"
        self._next_id += 1
        rec = {
            "account_id": aid,
            "account_type": kind,
            "open_day": kw.pop("open_day", int(self.rng.integers(-2500, -90))),
            "device_id": kw.pop("device_id", None),
            "is_mule": kw.pop("is_mule", 0),
            "mule_network_id": kw.pop("mule_network_id", ""),
            "typology": kw.pop("typology", ""),
            "is_exit": kw.pop("is_exit", 0),
            "cluster": kw.pop("cluster", -1),
        }
        rec.update(kw)
        self.accounts.append(rec)
        self._idx[aid] = rec          # keep index live as accounts are created
        return aid

    def _new_device(self) -> str:
        d = f"D{self._next_device:06d}"
        self._next_device += 1
        return d

    def _tx(self, src, dst, amount, minute, device):
        # clamp into the observation window
        if minute < 0 or minute >= self.T:
            return
        self.txns.append((int(minute), src, dst, round(float(amount), 2), device))

    # ---- populations ------------------------------------------------------
    def build_population(self):
        c, rng = self.cfg, self.rng

        self.employers = [self._new_account("employer") for _ in range(c.n_employer)]
        # exit points are laundering intermediaries too -- labelled is_mule so
        # the precision metric stays coherent, and is_exit so the recovery
        # replay can tell "still in the system" from "gone".
        self.exit_points = [self._new_account("cashout_point", is_mule=1,
                                              is_exit=1, typology="cashout")
                            for _ in range(c.n_exit_points)]
        self.merchants = [self._new_account("merchant") for _ in range(c.n_merchant)]
        # merchant popularity is power-law
        w = rng.pareto(1.4, size=len(self.merchants)) + 1.0
        self.merchant_w = w / w.sum()

        self.personal: list[str] = []
        for kind, n in (("salaried", c.n_salaried),
                        ("gig", c.n_gig),
                        ("student", c.n_student)):
            for _ in range(n):
                self.personal.append(self._new_account(kind))

        # confounders
        self.small_shops = [self._new_account("small_shop") for _ in range(c.n_small_shop)]
        self.group_collectors = [self._new_account("group_collector")
                                 for _ in range(c.n_group_collector)]
        self.gig_aggregators = [self._new_account("gig_aggregator")
                                for _ in range(c.n_gig_aggregator)]
        self.informal_lenders = [self._new_account("informal_lender")
                                 for _ in range(c.n_informal_lender)]

        self._assign_clusters()
        self._assign_devices()
        self._assign_peers()

    def _assign_clusters(self):
        """Legit accounts belong to social clusters, so the graph contains
        genuine dense communities that are NOT mule networks."""
        rng = self.rng
        pool = self.personal + self.small_shops + self.group_collectors
        rng.shuffle(pool)
        idx = {a["account_id"]: a for a in self.accounts}
        for i, aid in enumerate(pool):
            idx[aid]["cluster"] = i // self.cfg.cluster_size
        self.n_clusters = (len(pool) // self.cfg.cluster_size) + 1
        self.cluster_members: dict[int, list[str]] = {}
        for aid in pool:
            self.cluster_members.setdefault(idx[aid]["cluster"], []).append(aid)

    def _assign_devices(self):
        rng = self.rng
        idx = {a["account_id"]: a for a in self.accounts}
        legit = (self.personal + self.small_shops + self.group_collectors
                 + self.gig_aggregators + self.informal_lenders)
        i = 0
        while i < len(legit):
            dev = self._new_device()
            # occasional household sharing: 2-3 accounts on one handset
            if rng.random() < self.cfg.p_shared_household_device:
                k = int(rng.integers(2, 4))
            else:
                k = 1
            for aid in legit[i:i + k]:
                idx[aid]["device_id"] = dev
            i += k
        for aid in self.merchants + self.employers + self.exit_points:
            idx[aid]["device_id"] = self._new_device()

    def _assign_peers(self):
        """Stable, mostly-reciprocal peer sets, drawn within social cluster."""
        rng, c = self.rng, self.cfg
        idx = {a["account_id"]: a for a in self.accounts}
        self.peers: dict[str, list[str]] = {}
        for aid in self.personal:
            cl = idx[aid]["cluster"]
            local = [x for x in self.cluster_members.get(cl, []) if x != aid]
            k = int(rng.integers(c.peers_min, c.peers_max + 1))
            chosen = []
            for _ in range(k):
                if local and rng.random() > c.p_peer_outside_cluster:
                    chosen.append(local[int(rng.integers(len(local)))])
                else:
                    chosen.append(self.personal[int(rng.integers(len(self.personal)))])
            self.peers[aid] = list(dict.fromkeys(chosen))

    # ---- legitimate activity ---------------------------------------------
    def _personal_activity(self, aid, kind, intensity=1.0):
        """Ordinary account behaviour. Reused for mule accounts, which are
        real people's accounts and therefore carry genuine cover traffic."""
        rng, c = self.rng, self.cfg
        idx = self._idx
        dev = idx[aid]["device_id"]

        if kind == "salaried":
            emp = self.employers[int(rng.integers(len(self.employers)))]
            sal = float(rng.uniform(18_000, 95_000))
            payday = int(rng.integers(1, 6))
            for d in range(payday, c.days, 30):
                self._tx(emp, aid, sal * rng.uniform(0.97, 1.03),
                         d * MIN_PER_DAY + rng.integers(540, 1200),
                         idx[emp]["device_id"])
            budget = sal / 30.0
        elif kind == "gig":
            plat = self.employers[int(rng.integers(len(self.employers)))]
            for d in range(c.days):
                if rng.random() < 0.75 * intensity:
                    self._tx(plat, aid, rng.uniform(250, 2400),
                             d * MIN_PER_DAY + rng.integers(1080, 1380),
                             idx[plat]["device_id"])
            budget = 700.0
        else:  # student
            parent = self.personal[int(rng.integers(len(self.personal)))]
            for d in range(int(rng.integers(1, 6)), c.days, 30):
                self._tx(parent, aid, rng.uniform(4_000, 20_000),
                         d * MIN_PER_DAY + rng.integers(600, 1300),
                         idx[parent]["device_id"])
            budget = 350.0

        # merchant spend -- scaled so real accounts RETAIN part of inflow
        n_spend = int(rng.poisson(1.9 * c.days * intensity))
        if n_spend:
            targets = rng.choice(len(self.merchants), size=n_spend, p=self.merchant_w)
            for t in targets:
                minute = int(rng.integers(0, self.T))
                minute = (minute // MIN_PER_DAY) * MIN_PER_DAY + int(rng.integers(420, 1350))
                self._tx(aid, self.merchants[t],
                         max(20, rng.gamma(2.0, budget * 0.20)), minute, dev)

        # P2P with a stable, reciprocal peer set
        for p in self.peers.get(aid, []):
            n = int(rng.poisson(0.10 * c.days * intensity))
            for _ in range(n):
                m = int(rng.integers(0, self.T))
                amt = max(30, rng.gamma(2.0, 700))
                self._tx(aid, p, amt, m, dev)
                if rng.random() < 0.55:   # real people pay each other back
                    self._tx(p, aid, amt * rng.uniform(0.4, 1.6),
                             m + int(rng.integers(120, 6 * MIN_PER_DAY)),
                             idx[p]["device_id"])

    def gen_legit(self):
        rng, c = self.rng, self.cfg
        for aid in self.personal:
            self._personal_activity(aid, self._idx[aid]["account_type"])
        for mid in self.merchants:
            for d in range(0, c.days, 7):
                self._tx(mid, self.employers[int(rng.integers(len(self.employers)))],
                         rng.uniform(5_000, 80_000),
                         d * MIN_PER_DAY + rng.integers(1100, 1300),
                         self._idx[mid]["device_id"])

    # ---- confounders: legitimate, but mule-shaped -------------------------
    def gen_confounders(self):
        """The hard negatives. Any model that scores these top-200 is broken."""
        rng, c = self.rng, self.cfg
        idx = self._idx

        # 1. Small shop: enormous fan-in from strangers, low reciprocity.
        for aid in self.small_shops:
            dev = idx[aid]["device_id"]
            supplier = self.employers[int(rng.integers(len(self.employers)))]
            for d in range(c.days):
                n_cust = int(rng.poisson(rng.uniform(6, 26)))
                for _ in range(n_cust):
                    cust = self.personal[int(rng.integers(len(self.personal)))]
                    self._tx(cust, aid, max(20, rng.gamma(1.8, 160)),
                             d * MIN_PER_DAY + int(rng.integers(420, 1320)),
                             idx[cust]["device_id"])
                if d % 3 == 0:  # sweeps to supplier -- high pass-through
                    self._tx(aid, supplier, rng.uniform(3_000, 26_000),
                             d * MIN_PER_DAY + int(rng.integers(1200, 1400)), dev)

        # 2. Group collector: rent / chit fund / trip pool.
        #    pass-through ~1.0 and dwell measured in hours. Looks exactly
        #    like a mule collector on account-level features.
        for aid in self.group_collectors:
            dev = idx[aid]["device_id"]
            cl = idx[aid]["cluster"]
            members = [x for x in self.cluster_members.get(cl, []) if x != aid][:14]
            if not members:
                continue
            landlord = self.merchants[int(rng.integers(len(self.merchants)))]
            for d in range(2, c.days, int(rng.integers(7, 31))):
                total = 0.0
                base = d * MIN_PER_DAY + int(rng.integers(540, 1080))
                for m in members:
                    amt = rng.uniform(1_500, 9_000)
                    total += amt
                    self._tx(m, aid, amt, base + int(rng.integers(0, 600)),
                             idx[m]["device_id"])
                # forwards nearly everything, within hours
                self._tx(aid, landlord, total * rng.uniform(0.95, 1.0),
                         base + 600 + int(rng.integers(30, 900)), dev)

        # 3. Gig aggregator / fleet owner: high velocity, many counterparties.
        for aid in self.gig_aggregators:
            dev = idx[aid]["device_id"]
            riders = [self.personal[int(rng.integers(len(self.personal)))] for _ in range(20)]
            plat = self.employers[int(rng.integers(len(self.employers)))]
            for d in range(c.days):
                self._tx(plat, aid, rng.uniform(8_000, 40_000),
                         d * MIN_PER_DAY + int(rng.integers(600, 800)),
                         idx[plat]["device_id"])
                for r in riders:
                    if rng.random() < 0.5:
                        self._tx(aid, r, rng.uniform(400, 1800),
                                 d * MIN_PER_DAY + int(rng.integers(810, 1200)), dev)

        # 4. Informal lender / p2p trader: fast in-out, many uniques.
        for aid in self.informal_lenders:
            dev = idx[aid]["device_id"]
            for d in range(c.days):
                n = int(rng.poisson(3.5))
                for _ in range(n):
                    src = self.personal[int(rng.integers(len(self.personal)))]
                    dst = self.personal[int(rng.integers(len(self.personal)))]
                    amt = rng.uniform(2_000, 30_000)
                    t0 = d * MIN_PER_DAY + int(rng.integers(480, 1200))
                    self._tx(src, aid, amt, t0, idx[src]["device_id"])
                    self._tx(aid, dst, amt * rng.uniform(0.9, 0.99),
                             t0 + int(rng.integers(5, 240)), dev)

    # ---- mule networks ----------------------------------------------------
    def gen_mule_networks(self):
        rng, c = self.rng, self.cfg
        idx = self._idx
        typologies = list(c.typology_mix.keys())
        probs = np.array([c.typology_mix[t] for t in typologies], dtype=float)
        probs /= probs.sum()

        self.victims: list[str] = []
        all_mules: list[str] = []

        for net in range(c.n_mule_networks):
            typ = typologies[int(rng.choice(len(typologies), p=probs))]
            nid = f"NET{net:02d}"
            n_mules = int(rng.integers(*c.mules_per_network))

            # device sharing: only some networks are sloppy about it
            shares = rng.random() < c.p_network_shares_device
            devs = ([self._new_device()
                     for _ in range(int(rng.integers(*c.devices_per_network)))]
                    if shares else None)

            mules = []
            # reuse of mules across networks
            n_reused = int(len(all_mules) * 0) if not all_mules else \
                int(min(len(all_mules), rng.binomial(n_mules, c.mule_reuse_rate)))
            if n_reused:
                mules += list(rng.choice(all_mules, size=n_reused, replace=False))
            for _ in range(n_mules - len(mules)):
                if rng.random() < c.p_mule_is_aged_dormant:
                    open_day = int(rng.integers(-2000, -400))   # dormant, reactivated
                else:
                    open_day = int(rng.integers(-70, 5))        # freshly opened
                aid = self._new_account(
                    "mule", open_day=open_day, is_mule=1,
                    mule_network_id=nid, typology=typ,
                    device_id=(devs[int(rng.integers(len(devs)))] if devs
                               else self._new_device()),
                )
                mules.append(aid)
            for m in mules:
                if not idx[m].get("_covered"):
                    # a mule is a REAL person's account: give it genuine history
                    cover_kind = ["student", "gig", "salaried"][
                        int(rng.choice(3, p=[0.5, 0.35, 0.15]))]
                    self.peers[m] = [self.personal[int(rng.integers(len(self.personal)))]
                                     for _ in range(int(rng.integers(2, 6)))]
                    self._personal_activity(m, cover_kind,
                                            intensity=float(rng.uniform(0.7, 1.25)))
                    idx[m]["_covered"] = True
                idx[m]["is_mule"] = 1
                if nid not in idx[m]["mule_network_id"]:
                    idx[m]["mule_network_id"] = (
                        idx[m]["mule_network_id"] + "|" + nid).strip("|")
            all_mules = list(dict.fromkeys(all_mules + mules))

            self._cur_exits = list(rng.choice(
                self.exit_points,
                size=int(rng.integers(*c.exits_per_network)), replace=False))
            collectors = mules[:max(1, len(mules) // 10)]
            cashouts = mules[-max(1, len(mules) // 8):]

            for _ in range(int(rng.integers(*c.payouts_per_network))):
                victim = self.personal[int(rng.integers(len(self.personal)))]
                self.victims.append(victim)
                amount = float(rng.integers(*c.payout_amount))
                t0 = int(rng.integers(0, self.T - MIN_PER_DAY))
                collector = collectors[int(rng.integers(len(collectors)))]
                self._tx(victim, collector, amount, t0, idx[victim]["device_id"])
                self._layer(typ, collector, mules, cashouts, amount, t0, idx)

        self.mule_set = set(all_mules)

    def _cash_out(self, src, amount, t, idx):
        """The money leaves the banking system here."""
        rng = self.rng
        ex = self._cur_exits[int(rng.integers(len(self._cur_exits)))]
        self._tx(src, ex, amount * rng.uniform(*self.cfg.exit_share),
                 t + self._dwell(), idx[src]["device_id"])

    def _dwell(self) -> int:
        lo, hi = self.cfg.dwell_seconds
        return max(1, int(self.rng.uniform(lo, hi) / 60))  # minutes

    def _layer(self, typ, collector, mules, cashouts, amount, t0, idx):
        """Execute one laundering run. All hops are time-respecting."""
        rng, c = self.rng, self.cfg
        depth = int(rng.integers(*c.layering_depth))

        def cut(a):
            return a * (1.0 - rng.uniform(*c.cut_pct))

        if typ == "fan_out_star":
            t = t0 + self._dwell()
            legs = [m for m in mules if m != collector]
            k = min(len(legs), int(rng.integers(5, 13)))
            picks = rng.choice(legs, size=k, replace=False)
            share = cut(amount) / k
            for m in picks:
                tm = t + int(rng.integers(0, 6))
                self._tx(collector, m, share * rng.uniform(0.85, 1.15),
                         tm, idx[collector]["device_id"])
                co = cashouts[int(rng.integers(len(cashouts)))]
                if co != m:
                    t_co = tm + self._dwell()
                    self._tx(m, co, cut(share), t_co, idx[m]["device_id"])
                    self._cash_out(co, cut(share), t_co, idx)

        elif typ == "peel_chain":
            cur, amt, t = collector, amount, t0
            for _ in range(depth + 2):
                nxt = mules[int(rng.integers(len(mules)))]
                if nxt == cur:
                    continue
                t += self._dwell()
                amt = cut(amt)
                self._tx(cur, nxt, amt, t, idx[cur]["device_id"])
                cur = nxt
            self._cash_out(cur, amt, t, idx)

        elif typ == "round_trip_cycle":
            ring = list(rng.choice(mules, size=min(len(mules), depth + 3),
                                   replace=False))
            amt, t = amount, t0
            for i in range(len(ring)):
                src, dst = ring[i], ring[(i + 1) % len(ring)]
                if src == dst:
                    continue
                t += self._dwell()
                amt = cut(amt)
                self._tx(src, dst, amt, t, idx[src]["device_id"])
            self._cash_out(ring[-1], amt, t, idx)

        elif typ == "smurfing":
            t = t0 + self._dwell()
            n = int(rng.integers(12, 40))
            share = cut(amount) / n
            for _ in range(n):
                m = mules[int(rng.integers(len(mules)))]
                co = cashouts[int(rng.integers(len(cashouts)))]
                tm = t + int(rng.integers(0, 90))
                # deliberately sub-threshold amounts
                self._tx(collector, m, min(share, rng.uniform(4_000, 9_500)),
                         tm, idx[collector]["device_id"])
                if co != m:
                    amt_h = min(share, rng.uniform(4_000, 9_400)) * 0.97
                    t_co = tm + self._dwell()
                    self._tx(m, co, amt_h, t_co, idx[m]["device_id"])
                    self._cash_out(co, amt_h, t_co, idx)

    # ---- assembly ---------------------------------------------------------
    def run(self):
        self.build_population()
        self.gen_legit()
        self.gen_confounders()
        self.gen_mule_networks()

        tx = pd.DataFrame(self.txns,
                          columns=["minute", "sender", "receiver", "amount", "device_id"])
        tx = tx[tx.sender != tx.receiver].copy()
        tx = tx.sort_values("minute", kind="mergesort").reset_index(drop=True)
        tx.insert(0, "txn_id", [f"T{i:08d}" for i in range(len(tx))])
        t0 = pd.Timestamp("2026-01-01")
        tx["ts"] = t0 + pd.to_timedelta(tx["minute"], unit="m")

        for a in self.accounts:
            a.pop("_covered", None)
        acc = pd.DataFrame(self.accounts)
        # UPI distinguishes P2P from P2M. Registered merchants and employers
        # are P2M/payroll rails; everything else (incl. kirana shops running on
        # a PERSONAL VPA) is P2P and stays in the laundering graph.
        m2 = set(acc.loc[acc.account_type.isin(["merchant", "employer"]),
                         "account_id"])
        tx["channel"] = np.where(tx.sender.isin(m2) | tx.receiver.isin(m2),
                                 "P2M", "P2P")
        acc = acc.copy()
        acc["open_date"] = t0 + pd.to_timedelta(acc["open_day"], unit="D")
        acc = self._expose_labels(acc)
        return tx, acc

    def _expose_labels(self, acc: pd.DataFrame) -> pd.DataFrame:
        """Only a small, delayed subset of true mules is ever 'reported'."""
        rng, c = self.rng, self.cfg
        acc["label_exposed"] = 0
        acc["label_delay_days"] = np.nan
        mules = acc.index[acc.is_mule == 1]
        k = int(len(mules) * c.label_exposure_rate)
        if k:
            chosen = rng.choice(mules, size=k, replace=False)
            acc.loc[chosen, "label_exposed"] = 1
            acc.loc[chosen, "label_delay_days"] = rng.integers(
                c.report_delay_days[0], c.report_delay_days[1], size=k)
        return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--days", type=int, default=45)
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiply legit population; lowers the mule base rate")
    p.add_argument("--out", default="data")
    a = p.parse_args()

    cfg = Config(seed=a.seed, days=a.days, scale=a.scale)
    tx, acc = Simulator(cfg).run()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(out / "transactions.parquet", index=False)
    acc.to_parquet(out / "accounts.parquet", index=False)

    print(f"transactions : {len(tx):,}")
    print(f"accounts     : {len(acc):,}")
    print(f"true mules   : {int(acc.is_mule.sum()):,} "
          f"({acc.is_mule.mean()*100:.2f}% base rate)")
    print(f"labels shown : {int(acc.label_exposed.sum()):,}")
    print(f"confounders  : {int(acc.account_type.isin(['small_shop','group_collector','gig_aggregator','informal_lender']).sum()):,}")
    print(f"exit points  : {int(acc.is_exit.sum())}  "
          f"(value cashed out: Rs {tx[tx.receiver.isin(acc.loc[acc.is_exit==1,'account_id'])].amount.sum():,.0f})")
    print(f"P2P / P2M    : {(tx.channel=='P2P').sum():,} / {(tx.channel=='P2M').sum():,}")
    print(f"window       : {tx.ts.min()} -> {tx.ts.max()}")


if __name__ == "__main__":
    main()
