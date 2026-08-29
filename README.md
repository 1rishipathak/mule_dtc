# Network-Level Mule Detection — Week 1 Foundation

Working end-to-end pipeline: simulator → features → three detectors → Precision@k.

```bash
pip install numpy pandas networkx scikit-learn lightgbm python-louvain pyarrow

python simulate.py        --scale 6 --out data   # ~50s, 5.4M txns, 39k accounts
python features.py        --data data            # ~3 min
python laundering_graph.py --data data           # stage 4, ~4 min
python motifs.py          --data data            # stage 5, ~1 min
python model.py           --data data --k 200
python response.py        --data data            # stage 6, ~6 min
python alerts.py          --data data            # cross-bank payloads
python sweep.py --scale 4 --seed 13              # sensitivity harness
```

`--scale` multiplies the **legitimate** population only, so it directly controls
the mule base rate. `--scale 1` (~6.7k accounts, 3.4% base rate) is the fast dev
loop; `--scale 6` (~39k accounts, 0.54% base rate) is the number you report.

---

## Headline results (base rate 0.54%, 212 mules, 31 exposed labels, k=200)

| Detector | seed 42 (tuned) | seed 7 (held out) | Nets /14 | Conf. FP |
|---|---|---|---|---|
| Rules baseline | 0.075 | 0.105 | 14 / 11 | ~60% |
| Unsupervised, account-level | 0.245 | **0.005** | 12 / 1 | ~55% |
| GBM, sparse labels | 0.175 | 0.155 | 13 / 12 | ~2% |
| Control: activity only | 0.400 | 0.565 | 14 | ~3% |
| GBM, 8 curated | 0.565 | 0.470 | 14 | ~5% |
| Community score, no labels | 0.855 | 0.800 | 13 / 14 | 0% |
| **FINAL: community + chain depth** | **0.985** | **0.880** | **14** | **0%** |

Random selection scores 0.006. **Quote the held-out column** — every parameter
was tuned on seed 42. Report **lift** (~170x), not raw precision.

**The label-free network detector beats every supervised model.** That is the
project thesis, demonstrated rather than asserted: the network is the right unit
of analysis, and once you score networks you no longer need the labels you do not
have.

---

## Four things the build has already established

**1. Account-level aggregates do not work — and the simulator proves why.**
Mule accounts belong to real people, so they carry real cover traffic. Once that
cover traffic is present, the mule median dwell time is 181 minutes, sitting
*between* students (240) and gig aggregators (146). Aggregate pass-through ratio
has an AUC of 0.502 — literally coin-flip. Every behavioural feature is therefore
computed over a short **window** and reduced with an extreme statistic
(min / p10 / peak), never a mean. This is the single most important design
decision in the codebase and it is a strong pitch point.

**2. The confounder population is doing its job.** The simulator includes
legitimate accounts that are deliberately mule-shaped: kirana shops on personal
VPAs (huge fan-in, no reciprocity), rent/chit-fund collectors (pass-through
genuinely ~1.0, dwell in hours), gig fleet aggregators, informal lenders.
**60% of the rules baseline's alerts are these legitimate accounts.** That single
number is your best slide: it shows precisely how account-level scoring fails and
why your precision figure means something.

**3. Supervised learning is starving, exactly as predicted.** With 31 exposed
labels, the 27-feature GBM (0.350) is *beaten by* a control model using only
volume and count features (0.490), and a hand-curated 8-feature set beats both
(0.540). The model is overfitting the label set, not learning laundering. Feature
importance shows only ~31 total tree splits.

**4. Adding the unsupervised score as a GBM feature made things worse**
(0.155). The two-stage design in the problem statement needs rethinking — the
stages currently fight each other.

---

## Honesty checks built into `model.py`

Keep both of these. They are cheap and they pre-empt the two questions a sharp
judge will ask.

- **Activity-only control.** A model using only volume/count features. If the real
  model cannot beat it, the system is detecting inactive accounts, not mules.
  This check caught a genuine artifact during the build: mule cover traffic was
  initially generated at lower intensity than legitimate traffic, which pushed
  `n_txn` to 0.949 AUC. Fixed by drawing mule activity from the same
  distribution as everyone else.
- **Confounder audit.** What fraction of each detector's alerts are legitimate
  shops and collectors.

---

## Design decisions worth defending in the pitch

- **P2P/P2M split.** UPI flags merchant transactions separately, so the laundering
  graph is built on P2P rails only. This cut the graph from 3.5M edges to 416k
  and is what a real deployment would do. Kirana shops on *personal* VPAs stay in
  the graph — which is why they remain hard negatives.
- **Time-respecting paths.** `time_resp_dist_to_seed` does forward BFS with
  non-decreasing timestamps, so structurally-present but temporally-impossible
  chains are excluded.
- **Labels are exposed, not given.** Ground truth exists by construction, but only
  15% of mules are ever "reported", with a 3–11 day delay. Unexposed mules sit in
  training labelled 0 — the positive-unlabelled situation a real bank faces.
- **Out-of-fold scoring** throughout, so no detector is evaluated on its own
  training rows.

---

## Stage 4: the laundering subgraph (`laundering_graph.py`)

Running community detection on all P2P transfers returns *the mule crew plus
everyone they know* — a mule's ordinary friends get pulled in by cover traffic.
Measured: three communities held 204 of 212 mules but were only ~35% mule, which
caps precision near 0.35.

The fix is that **the laundering graph is not the transfer graph.** Keep only
edges `u -> v` where `v` forwarded a comparable amount onward fast — the
operational definition of a layering hop, time-respecting by construction.
Ordinary people keep what they receive, so cover traffic drops out.

Tuned by sweep to a 15-minute window, 80% of the credit forwarded, ₹5,000 floor:
- 4,307 edges survive out of 2.1M P2P transfers (0.2%)
- 3,851 accounts remain, holding 188 of 212 mules — **a 9x enrichment before any
  model runs**
- mule purity of mule-containing communities rose from 0.32 to **0.67**

The ₹5,000 floor is deliberate and worth defending out loud: a ₹20,000 floor
gives a visibly cleaner graph, but the smurfing typology uses sub-threshold
₹4,000–9,500 transfers, so the cleaner graph is bought by silently deleting an
entire typology.

The community score itself is a transparent weighted sum of z-scored community
features (low dwell, low reciprocity, shared devices, fast flow-through, new
accounts). Not a black box — which matters, because this score throttles accounts.

## Stage 5: motifs (`motifs.py`)

Four typology fingerprints, computed on the 4k-edge laundering subgraph, so all
of it is cheap. Cycles are counted as closed walks via sparse matrix powers
rather than enumerating simple cycles, which does not terminate reliably.

Measured within the 3,851 candidate accounts (base rate 4.9%):

| Motif | AUC | Verdict |
|---|---|---|
| `motif_chain_depth` | **0.978** | strongest feature in the project |
| `lg_val_in` | 0.943 | keep |
| `lg_in_deg` / `lg_in_uniq` | 0.90 | keep |
| `motif_smurf_share` | 0.655 | weak |
| `motif_star_out` | 0.610 | weak |
| `motif_cycle2/3/4` | 0.54 | near-useless — only ~15% of networks are cyclic |

Chain depth is the longest time-respecting path ending at a node. Legit
flow-through bottoms out at depth 2 (a rent collector receives, then forwards);
layering chains run 3-6 deep.

**Apply chain depth per-account, never aggregated.** It describes a node's
POSITION in a chain, not a property of a group. Folding it into the community
table dropped the score from 0.915 to 0.790.

## The supervised ranker: demoted, on the evidence

With ~31 exposed labels every supervised variant loses to the label-free
detector: full GBM 0.155, activity-only control 0.490, curated 8-feature GBM
0.540, curated + motifs 0.755 — against 0.915 for the unsupervised system.

Present the label-free network detector AS the system, and the GBM honestly as a
refinement that needs more labels than any demo can supply. That is both true and
a stronger story than a mediocre classifier: it means the approach works in
exactly the situation banks are actually in, where confirmed labels barely exist.

## Stage 6: graded response, evaluated causally (`response.py`)

Every number above this section is computed over the full 45-day window, so the
features can see the future. Fine for feature research, fatal for a real-time
claim. Stage 6 rebuilds the score from quantities computable from the past only
(chain-depth DP in time order, running in-degree and in-value, device sharing
known at KYC). Louvain stays out of the streaming path: the partition is
unstable early in a network's life.

**Lookahead is worth 0.235.** Causal streaming precision@200 = **0.750**, against
0.985 for the batch ranking on the same data. That gap is the honest price of
real time, and it widened when chain depth was burst-scoped — the fix that made
the evidence truthful also removed a chunk of accidental lookahead.

Two different claims, two different numbers, and they must not be mixed:

| Claim | Number |
|---|---|
| "Our ranking finds mule networks" (batch, full window) | 0.985 tuned / **0.880 held out** |
| "Operating in real time on past data only" | **0.750** |
| "The accounts we would actually act on" (causal freeze tier, top 154) | **0.903**, 15 innocent |

**Counterfactual recovery replay**, blocking at the freeze threshold:

(Superseded by the four-seed table below; the buckets partition the payout
exactly in every run.) An earlier version double-counted:
blocked funds were not removed from the movable pool, so the same rupee was
blocked once per attempted transfer and "recovery" exceeded 100%.

### Two findings that contradict the original design

**The score is bimodal, so four tiers are not supported by the data.** The top
171 accounts are 90.6% mule; below that, precision falls off a cliff and no band
of 10+ accounts clears even 0.55. A `hold` tier between freeze and throttle
cannot be calibrated because there is nothing in the middle to put in it. Present
two tiers honestly rather than inventing a third.

Tiers must also be calibrated on precision **within the band**, not cumulative
precision from rank 0 — an account in the hold band is never seen by the freeze
rule. The cumulative version delivered 0.315 against a 0.70 target.

**Cash-out is now modelled** (`is_exit` accounts: ATM agents, crypto off-ramps,
complicit shopfronts). ~Rs 29-30M genuinely leaves the banking system per run, so
the recovery counter finally has a real "gone" side.

### The terminal-hop blind spot

The flow-through test asks whether the RECEIVER forwarded money onward, so it
structurally **cannot see the last node in a chain** — exactly the cash-out point
where funds exit. Measured: all 14 exit accounts had in-degree 0 and were absent
from the graph entirely, while overall precision looked fine at 0.965. The
detector was catching the mules and missing the money.

Fix: a chain includes its own endpoint. A transfer out of an identified
laundering node, above the value floor, **within the same burst** as the credit
that funded it, is kept even when the receiver never forwards. The timing
constraint is essential — without it, 42,292 spurious edges appear and the graph
grows from 3.8k to 26.5k nodes, erasing the enrichment. With it: 3,971 terminal
edges, and 8 of 14 exit points enter the top 200.

### Counterfactual recovery, with a real exit

Four seeds, full pipeline re-derived each time:

| seed | 42 | 7 | 13 | 21 |
|---|---|---|---|---|
| **Intercepted** | 76.3% | **53.9%** | 77.9% | 74.4% |
| Cashed out — gone | 5.2% | 7.5% | 2.8% | 3.1% |
| Still in the mule layer | 18.4% | 38.7% | 19.3% | 22.4% |

Buckets partition the payout exactly in every run.

**Quote "roughly 70%, ranging 54-78% across seeds."** Mean 70.6, median 75.4,
sd 11.2 — a far wider spread than precision (0.95 ± 0.07), because recovery
depends on *when* an account crosses the freeze threshold relative to when the
money moves, not just on ranking quality.

Seed 7 is a genuine outlier and **it is not explained**. The obvious hypothesis
— that recovery suffers when networks debut after the watchlist is drawn — fails:
seed 7 has the *fewest* previously-unseen receiving accounts (0.5%, against 21%
for seed 42) alongside the *worst* recovery, the opposite of the prediction. With
n=4 no correlation is meaningful. If asked, say it is unexplained dispersion and
that the honest figure is the range.

## Sensitivity: the two questions a judge will ask (`sweep.py`, `sensitivity.png`)

**"Isn't your dataset easier than reality?"** Precision holds as the base rate
falls toward the real ~0.1%, while both baselines degrade:

| Accounts | Base rate | Network detector | GBM | Rules | Lift |
|---|---|---|---|---|---|
| 6,763 | 3.59% | 0.995 | 0.910 | 0.475 | 28x |
| 13,246 | 2.01% | 0.880 | 0.875 | 0.325 | 44x |
| 26,178 | 1.06% | 1.000 | 0.820 | 0.200 | 94x |
| 39,055 | 0.60% | 0.985 | 0.565 | 0.075 | 164x |
| 52,000 | 0.50% | 0.990 | 0.555 | 0.085 | 198x |
| 64,936 | **0.43%** | **0.990** | 0.655 | **0.040** | **233x** |

Lift climbs from 28x to **233x** as the problem gets harder. That is the shape
you want: the rules baseline and the supervised model both fall apart at a
realistic base rate; the network detector does not. 14/14 networks at every
scale.

**"Is that one lucky seed?"** Four seeds at scale 4, full pipeline re-derived
each time with nothing cached:

| seed | 42 | 7 | 13 | 21 |
|---|---|---|---|---|
| precision@200 | 1.00 | 0.85 | 1.00 | 0.95 |

**0.95 ± 0.07**, range 0.85-1.00, 14/14 networks every run, confounder false
alerts never above 1.5%. Quote the mean and the spread, never the best run.

Not tested below 0.43%. Scale 16 (~104k accounts) is killed by the OOM killer
on 3GB; 64,936 accounts and ~9M transactions is the ceiling here. Your laptops
will go further. Real UPI is nearer 0.1%, so say "holds to 0.43%, untested
below" rather than extrapolating.

## The demo (`demo.html`)

A replay of a real episode from the pipeline, not a scripted animation: network
NET11's **debut** payout of Rs 3,59,571, layered across 11 accounts. Open it in a
browser, hit Play, or scrub the timeline by hand during questions.

| | |
|---|---|
| Accounts caught, account-level rules | **2 / 11** |
| Accounts caught, network view | **8 / 11** |
| First detection | **T+8 min**, before any fraud report exists |
| Value held | **Rs 2,23,963 (62.3%)** |
| Value still moving at T+45 | Rs 1,35,608 |
| Value cashed out | Rs 0 |

Everything reconciles with the pipeline: ranks come from `final_score`, and the
money split is a taint replay counting each rupee once (held + moving + gone
equals the payout exactly).

Two things were fixed to keep the demo honest, and both are worth knowing in
case anyone asks:

- **Detection time was originally global**, so accounts already flagged from
  earlier episodes appeared to be detected at T-57032 minutes. The demo now uses
  a network's *debut* payout, so nothing is known in advance.
- **Detection is timed to the confirming outbound transfer**, not the incoming
  credit. Flow-through is only *observable* once the account forwards; crediting
  detection to the arrival would be lookahead.

The cash-out bar stays at zero in this episode because nothing escapes before
the network is flagged. Say so rather than implying that is typical — across the
full dataset 4-8% gets out.

## The console (`console.html`)

The operator-facing view: what a bank analyst would actually work in. Populated
from the real `alerts.json`, not mock data.

- Alert queue with tier filter, and a detail pane showing the evidence behind
  each alert with percentile bars
- A **"reveal ground truth" toggle**. The queue deliberately includes **7 false
  positives** -- mostly informal lenders, plus one salaried worker and one
  student. An analyst UI that only shows correct answers is a lie, and the
  ambiguous cases are the actual job. The toggle exists only because this is
  simulated data; in deployment there is no such column, which is exactly why
  the detector uses no labels.
- Every alert states its basis and that no PII crosses the boundary

## Chain depth was reporting a falsehood

The depth DP had no reset, so separate laundering runs weeks apart concatenated
into one apparent path: mules were reported at 19 hops against a configured
layering depth of 2-4. Detection still worked — but the alert payload shipped
that sentence to another bank.

Depth is now burst-scoped (`CHAIN_GAP = 720` min) in both `motifs.py` and the
streaming scorer. Median mule depth is now 2, max 7. Held-out precision moved
0.900 -> 0.880: two points of precision traded for an explanation that is true.
That is the right trade when the output throttles someone's account.

## Cross-bank alert payload (`alerts.py`)

The obvious objection to the cross-institution claim is that banks cannot pool
customer data. So the payload carries the least information that still lets a
receiving bank act: account handle hashed with a per-window salt, no name, no
balance, no counterparty list, no transaction detail. Evidence is a list of
CODES with percentiles, so the receiver learns "sits 4 hops deep in a fast
forwarding chain", not "received Rs 4,80,000 at 14:32 from handle X".

That is the honest version of the argument: **derived risk signals cross the
boundary, customer data does not.**

An alert with no citable evidence is dropped rather than sent — 25 of 200 tiered
accounts produced no extreme signal and were suppressed. A bank cannot act on an
unexplained score, and RBI's FREE-AI principles put Explainability and
Accountability first.

Of the 16 innocent accounts in the freeze tier, all 16 are confounders — kirana
shops on personal VPAs and rent collectors. That is the honest cost of the
action, and it belongs on the slide.

## Status

Every build item is complete. What remains is not code:

1. **Ground the typologies in FIU-IND red-flag indicators and RBI circulars.**
   Still outstanding, and it is the answer to "aren't you just detecting the
   patterns you injected?" Half a day of reading; adjust `simulate.py` parameters
   if a typology does not match, then re-run one command per stage.
2. **Slides.** Strongest three: the sensitivity curve, the finding that 60% of
   baseline alerts land on kirana shops, and "what we do not claim".
3. **Rehearse, with the code frozen.**

## Known constraints

- Built and tested on 3GB RAM / 1 core. `--scale 6` fits; larger will need more.
- `features.py` loops over accounts in Python (~3 min at scale 6). Fine for a
  hackathon; would need rewriting for production.
- Typologies are currently my parameterization, **not** derived from published
  FIU-IND red-flag indicators. The problem statement flags this as the critical
  credibility move — it has not been done yet, and it should be, before the
  numbers go on a slide.
