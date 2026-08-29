# Presenter's guide

Everything you need to understand the system well enough to defend it, plus the
exact order to run the demo.

---

# Part 1 — What you built, in one paragraph

Indian banks score mule accounts one at a time, and a mule is invisible that way:
valid KYC, small balance, ordinary history. We stopped scoring accounts and
started scoring **networks**. We build a graph of only the transfers that look
like laundering hops, find tightly-connected groups inside it, and score the
group. It uses **no fraud labels at all**, which matters because real banks have
almost none. On unseen data it puts 88% true mules in its top 200 alerts, versus
a rules baseline where 60% of alerts are corner shops.

---

# Part 2 — How it works, stage by stage

## Stage 1 — The simulator (`simulate.py`)

Real UPI data is unobtainable outside a regulated entity, so we generate it.

**What's in the population (39,055 accounts at `--scale 6`):**
- Salaried workers, gig workers, students, merchants, employers
- **Confounders** — legitimate accounts deliberately shaped like mules: kirana
  shops on personal VPAs (huge fan-in, no reciprocity), rent and chit-fund
  collectors (pass-through genuinely ~1.0), gig fleet aggregators, informal
  lenders. About 2,160 of them.
- **235 mules** across **14 networks**, using four typologies: fan-out star,
  peel chain, round-trip cycle, smurfing
- **14 cash-out points** — ATM agents, crypto off-ramps — where money leaves the
  banking system

**The three design decisions that matter:**

1. **Mules carry real cover traffic.** A mule account belongs to a real student
   or worker, so it has real salary credits and real shopping. Without this, a
   model just learns "inactive account = mule."
2. **Only 35 labels are exposed**, delayed 3-11 days. The other 200 mules sit in
   training labelled 0 — the positive-unlabelled situation banks actually face.
3. **The confounders are the point.** They are why the precision number means
   anything.

**Why you can defend synthetic data:** the model never sees the labels. It is
scored on structure it discovers, not patterns it was told about.

## Stage 2 — Features (`features.py`)

**The single most important finding in the project:** once mules carry cover
traffic, *every account-level average stops working*. Median mule dwell time is
181 minutes — sitting between students (240) and gig aggregators (146).
Aggregate pass-through ratio has an AUC of **0.502**. That is a coin flip.

So every behavioural feature is computed over a short **window** and reduced with
an extreme statistic (min, 10th percentile, peak) — never a mean. The signal is
in the burst, and averaging over 45 days destroys it.

## Stage 3 — Baselines and honesty checks (`model.py`)

Three detectors to beat, plus two checks most projects skip:

- **Activity-only control** — a model using nothing but transaction counts and
  volumes. If the real model can't beat it, you're detecting inactive accounts.
  *This caught a genuine bug*: mule cover traffic was initially generated at
  lower intensity, pushing `n_txn` to 0.949 AUC.
- **Confounder audit** — what share of each detector's alerts are corner shops.

## Stage 4 — The laundering subgraph (`laundering_graph.py`) ⭐

**This is the core idea. Learn it properly.**

Running community detection on all P2P transfers returns *"the mule crew plus
everyone they know"* — a mule's ordinary friends get dragged in by cover traffic.
Measured: three communities held 204 of 212 mules but were only ~35% mule, which
caps precision around 0.35 no matter how good your scorer is.

**The fix: the laundering graph is not the transfer graph.** Keep only edges
`u -> v` where `v` forwarded a comparable amount onward *fast*. That is the
operational definition of a layering hop, and it is time-respecting by
construction. Ordinary people keep what they receive, so cover traffic evaporates.

Tuned by sweep: **15-minute window, 80% of the credit forwarded, ₹5,000 floor.**
- ~9,400 edges survive from 2.1M P2P transfers (0.4%)
- Mule purity of mule-containing communities: 0.32 → **0.81**

**Defend the ₹5,000 floor out loud.** A ₹20,000 floor gives a visibly cleaner
graph — bought by silently deleting the entire smurfing typology, which uses
sub-threshold ₹4,000-9,500 transfers.

Then: detect communities, score each community, broadcast the score to members.
The score is a transparent weighted sum of z-scored features (low dwell, low
reciprocity, shared devices, fast flow-through, new accounts). Not a black box —
which matters, because this throttles people's accounts.

## Stage 5 — Motifs (`motifs.py`)

Four typology fingerprints. Two worked, two didn't — say so.

| Motif | AUC within candidates | Verdict |
|---|---|---|
| **Chain depth** | **0.978** | strongest feature in the project |
| Laundering in-value / in-degree | 0.90-0.94 | keep |
| Smurfing share | 0.655 | weak |
| Star fan-out | 0.610 | weak |
| Cycles (2/3/4) | 0.54 | near-useless — only ~15% of networks are cyclic |

**Chain depth** = longest time-respecting path ending at an account. Legit
fast-forwarding bottoms out at depth 2 (a rent collector receives, forwards,
done). Layering chains run 3-6 deep. This is the peel-chain typology appearing as
pure graph structure, and it is the payoff of enforcing time-ordering.

**Apply it per-account, never averaged into the community table** — it describes
a node's *position*, and aggregating it dropped the score from 0.915 to 0.790.

**Final detector — two terms, no labels:**
```
score = z(community_score) + 1.5 × z(chain_depth)
```

## Stage 6 — Graded response (`response.py`, `alerts.py`)

Everything above uses the full 45-day window, so features can see the future.
Fine for research, fatal for a real-time claim. So this stage rebuilds the score
from **past-only** quantities and splits the timeline at day 30.

**Two findings that contradict the original plan:**

1. **Four tiers aren't supported by the data.** The score is bimodal: the top
   ~154 accounts are 90.3% mule, then precision falls off a cliff. No band of
   10+ accounts clears even 0.55, so a `hold` tier between freeze and throttle
   has nothing to put in it. Two tiers, honestly, beats four invented ones.
2. Tiers must be calibrated on precision **within the band**, not cumulative from
   rank 0. The cumulative version delivered 0.315 against a 0.70 target.

**The cross-bank payload** answers the hardest policy question. Banks can't pool
customer data — so the alert carries hashed account references and evidence
*codes* only. No name, no balance, no counterparty list, no transaction detail.
A receiving bank learns *"sits 4 hops deep in a fast-forwarding chain"*, never
*"received ₹4,80,000 at 14:32 from handle X"*. **Derived risk signals cross the
boundary; customer data does not.** Alerts with no citable evidence are
suppressed rather than sent — 25 of 200 were dropped.

---

# Part 3 — Your numbers (memorise these)

| Claim | Number |
|---|---|
| Batch ranking, held-out seed | **0.880** precision@200 |
| Across four seeds | **0.95 ± 0.07** |
| **Real-time (causal, past data only)** | **0.750** |
| **Accounts we'd actually act on** (causal freeze tier, top 154) | **0.903**, 15 innocent |
| Networks found | **14 of 14**, every run |
| Lift over random at 0.43% base rate | **233×** |
| Funds intercepted | **~70%**, range 54-78% across seeds |
| Rules baseline alerts that are corner shops | **60%** |
| Labels used | **zero** |

**Do not mix the batch and causal numbers.** "Our ranking finds mule networks" is
0.880. "Operating in real time" is 0.750. If you quote 0.880 while claiming real
time, and someone notices, you lose the room.

---

# Part 4 — The demo, in order

Total: ~6 minutes. Have all three files open in browser tabs **before** you start.

### Before you begin
- Tab 1: `demo.html` — reset to T+0
- Tab 2: `console.html` — ground-truth toggle **off**
- Tab 3: `sensitivity.png`
- Laptop on mains, notifications off, no live code execution

### 0:00-0:45 — The problem
> "Almost every rupee stolen in Indian digital fraud leaves through a mule
> account. It has valid KYC, a small balance, and no history of wrongdoing.
> Nothing about it triggers a rule — because the anomaly isn't the account, it's
> the shape of the network around it."

Do not open anything yet. Just say it.

### 0:45-2:15 — The demo (Tab 1)
Press **Play**. Narrate over it:

- *"A real ₹3.6 lakh payout, replayed from our pipeline — not a scripted animation."*
- At ~T+8: *"The structural score fires here. No fraud report exists yet. The
  victim doesn't know anything is wrong."*
- At the end: *"Account-level rules caught 2 of 11. The network view caught 8."*
- Point at the money bar: *"₹2.2 lakh of ₹3.6 lakh held — 62%."*

**Scrub back and forth** if someone asks a question. That's what the slider is for.

### 2:15-3:15 — How it works (one slide, no code)
> "The trick is that the laundering graph is not the transfer graph. We keep only
> transfers where the receiver forwarded the money onward within fifteen minutes.
> Ordinary people keep what they receive. That single filter takes 2.1 million
> transfers down to about nine thousand, and concentrates 220 of our 235 mules
> into it — a nine-fold enrichment before any model runs."

Then: *"We score the community, not the account. And we use zero fraud labels."*

### 3:15-4:15 — The console (Tab 2)
- *"This is what an analyst sees. Every alert carries the evidence that produced
  it — no explanation, no alert."*
- Click one, read an evidence line aloud.
- **Then flip the ground-truth toggle.** *"Seven of these are wrong — mostly
  informal lenders. We left them in, because the ambiguous cases are the actual
  job. And this toggle only exists because it's simulated data. In deployment
  there's no such column — which is exactly why we use no labels."*

This is your credibility moment. Don't rush it.

### 4:15-5:15 — Does it hold up? (Tab 3)
Point at the left chart:
> "As the base rate falls toward reality, the rules baseline collapses from 0.475
> to 0.04, and a supervised model collapses too. Ours stays flat. Lift goes from
> 28× to 233×. Right chart: four random seeds, 0.95 ± 0.07."

### 5:15-6:00 — The cost of being wrong
> "Mules are usually recruited victims — students paid ₹3,000. So the output is
> graded, not a freeze. And the alert that crosses to another bank carries hashed
> references and evidence codes only: no name, no balance, no transaction detail.
> Derived risk signals cross the boundary; customer data does not."

Close on: *"We don't claim production accuracy — this is simulated data. We don't
claim we solved cross-bank data sharing. We claim that if the network view
exists, this is what becomes possible."*

---

# Part 5 — Hard questions, honest answers

**"Aren't you just detecting the patterns you injected?"**
> The model never sees the labels — it's fully unsupervised, so it finds structure
> rather than patterns it was told about. And our typologies come from published
> FIU-IND red-flag indicators and RBI circulars, not our imagination.
*(Only say the second sentence once you've actually done that reading.)*

**"Your data is synthetic and easier than reality."**
> Probably, and we say so on the slide. What we can show is direction: as we make
> it harder — base rate from 3.6% down to 0.43% — the baselines collapse and ours
> holds. We couldn't test below 0.43% on our hardware. Real UPI is nearer 0.1%.

**"Is this real time?"**
> The honest number is 0.750, not 0.880. We split the timeline and rebuilt the
> score from past-only data. Lookahead was worth 0.235, and we report both.

**"What happens when you freeze an innocent person?"**
> It happens — 15 of the 154 accounts in our freeze tier. Almost all are kirana
> shops on personal VPAs and rent collectors. That's why the response is graded
> and why every alert carries its evidence. We'd rather show that number than
> hide it.

**"Why not a GNN / deep learning?"**
> We tried a gradient-boosted model on the labels we had. With 35 labels it lost
> to a control model using nothing but transaction counts. The label-free network
> score beat everything. A GNN would need far more labels than a real bank has.

**"How is this different from MuleHunter.AI?"**
> We're not claiming novelty in detecting mules. Our contribution is the unit of
> analysis — network rather than account — plus real-time graded action and a
> cross-institution payload that carries no customer data.

**"Why did your cycle motifs not work?"**
> Only about 15% of our networks use a cyclic typology, so there was very little
> to find — AUC 0.54. We kept the code and report it rather than quietly dropping
> the features that didn't pan out.

---

# Part 6 — Things not to say

- Don't say "99% accurate." At a 0.6% base rate, predicting "not a mule" for
  everything is 99.4% accurate and worthless. Say **precision@200** and **lift**.
- Don't quote 0.985 — that's the seed we tuned on. Quote **0.880** held out.
- Don't quote 76% recovery. Say **~70%, ranging 54-78%**.
- Don't claim the demo's zero-rupees-cashed-out is typical. Across the dataset
  4-8% gets out.
- Don't claim you solved cross-bank data sharing. You showed what becomes
  possible if the network view exists.

---

# Part 7 — Reproducing everything

```bash
pip install numpy pandas networkx scikit-learn lightgbm python-louvain pyarrow

python simulate.py         --scale 6 --out data    # ~50s
python features.py         --data data             # ~3 min
python laundering_graph.py --data data             # ~2 min
python motifs.py           --data data             # ~30s
python model.py            --data data --k 200     # the leaderboard
python response.py         --data data             # causal split + recovery
python alerts.py           --data data             # cross-bank payloads
python sweep.py --scale 4 --seed 13                # sensitivity
```

If a judge asks to see it run, run `model.py` — it's the shortest path to a
number on screen. **Never run anything live that you haven't run that morning.**
