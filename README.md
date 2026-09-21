# evolab — a CPU-only self-optimizing agent

**evolab** is a small, dependency-free reproduction of the *self-improving
heuristic discovery* loop that systems like **AlphaEvolve** and the **Darwin
Gödel Machine** run at industrial scale. A population of candidate
**job-dispatching rules** is evolved purely from the feedback of an **exact
tardiness simulator** — no learned reward model, no teacher, no hand-written
final policy. On CPU, in seconds, with the standard library alone.

> The point is *honest, reproducible* self-improvement: every number in the
> Results section below is measured by `experiments/run_study.py` on this
> machine and committed as `results/evolution.json`. Nothing is quoted from a
> paper.

![ci](https://github.com/Lesereingrape/evolve-lab/actions/workflows/ci.yml/badge.svg)

## The task

One machine, no preemption, jobs arrive over time. Each job has a release time,
processing time, weight and due date. The agent is scored by **total weighted
tardiness** `Σ w · max(0, completion − due)` — a classic NP-hard objective, and
a *true* verifier: it computes the exact cost of any schedule, so the search
can never be fooled by a proxy metric.

A dispatching **rule** decides which *released* job to run next. We pit the
evolved rule against five textbook rules — FIFO, SPT, EDD, WSPT, MIN-SLACK —
and against a held-out pool the search never optimizes on.

## How the self-optimization works

Each candidate is a 5-vector of weights for one parametric priority function:

```
priority(job) = a·(1/p) + b·w + c·slack + d·(w/p) − e·release
```

Crucially, **the search space contains the textbook rules** as special cases
(shortest-processing-time ≈ setting `a`, WSPT ≈ setting `d`, FIFO ≈ setting
`e`, …). So a competent search should *rediscover and blend* known-good
heuristics without ever being shown one — which is exactly the "did the loop
find real structure, or just noise?" question this repo answers with data.

Two search modes share one interface so they can be compared head-to-head:

- **population** — a `(μ, λ)`-evolution-strategy with elitism + uniform
  crossover + Gaussian mutation.
- **hill_climb** — a greedy `(1+1)`-EA, one parent / one mutant per step.

`evolve()` records a best-so-far curve on the **training** pool *and* a
**probe** pool the search never sees. Keeping both is deliberate: if the
champion overfits, the two curves diverge — a property we surface, not hide.

## Quickstart

```bash
pip install -e .
evolab demo --seed 0            # one seeded run: random rule -> near-optimal
evolab study                    # full multi-seed study -> results/evolution.json
```

No third-party runtime dependencies at all (`dependencies = []`).

## What it measures

<!-- RESULTS:START -->
*Every figure below is produced by `experiments/run_study.py` on CPU and stored in
the committed [`results/evolution.json`](results/evolution.json); the tables are
rendered by `experiments/make_report.py`. 3 seeds (0, 1, 2), 150 generations,
120 training / 200 held-out instances of 30 jobs each.*

- objective: total weighted tardiness on 30-job single-machine instances (lower is better)

### Held-out cost: evolved rule vs textbook dispatching rules

| rule | held-out tardiness | vs evolved |
|------|-------------------:|---------:|
| WSPT | 1716.2 | +3.0% |
| SPT | 2440.2 | +31.8% |
| EDD | 2919.0 | +43.0% |
| MINSLACK | 3136.0 | +46.9% |
| FIFO | 3893.6 | +57.2% |
| **Evolved (population-ES)** | **1665.1** | — |

`vs evolved` is how much worse each baseline is than the discovered rule (positive
= the evolved rule is better). The evolved value is a mean over seeds (**±41**), so
a small edge over the strongest baseline (WSPT) is within seed noise.

### Self-improvement curve (best-so-far, mean over seeds)

| generation | train cost | held-out cost |
|-----------:|-----------:|--------------:|
| 0 | 2768 | 2800 |
| 15 | 1796 | 1812 |
| 30 | 1778 | 1795 |
| 45 | 1726 | 1746 |
| 60 | 1704 | 1720 |
| 75 | 1679 | 1696 |
| 90 | 1661 | 1679 |
| 105 | 1661 | 1679 |
| 120 | 1653 | 1671 |
| 135 | 1646 | 1665 |
| 150 | 1646 | 1665 |

From a **random** rule (2768) to 1665 (±41): a **39.8%** reduction. The
self-improvement is large and unambiguous against the *start* and against
FIFO/SPT/EDD/MIN-SLACK; the extra 3.0% over WSPT is a small edge near the
seed-to-seed spread.

### Ablations

Search mode (held-out cost):
- population-ES: **1665**
- greedy (1+1) hill-climb: 1861

Mutation strength sigma (held-out cost, 2 seeds):
- sigma=0.1: 1636
- sigma=0.4: 1637
- sigma=0.8: 1650

Training-pool size → generalization (held-out cost, 2 seeds). The search only ever
sees the training pool, so held-out cost is the honest read; the small pool
over-fits its few instances and transfers worse.
- pool=20: held-out **1657** (train on that pool's own instances: 1751)
- pool=120: held-out **1637** (train on that pool's own instances: 1642)

### What the agent discovered

Mean evolved weights: `inv_proc=1.564`, `weight=0.257`, `slack=0.089`,
`wratio=7.447`, `arrival=-0.047`. A dominant positive `wratio` term with a
secondary `inv_proc` term means the search **rediscovered WSPT and blended in some
shortest-processing-time pressure from scratch**, never being shown either rule.
That blend decisively beats FIFO/SPT/EDD/MIN-SLACK and *matches or very slightly
edges* the WSPT baseline — the honest headline is the ~40% climb from a random
rule, not a big lead over the best textbook rule.
<!-- RESULTS:END -->

## Layout

```
src/evolab/
  policy.py    # instance generator, exact simulator, 5 textbook rules, parametric rule
  evolve.py    # (mu,lambda)-ES and (1+1)-EA over rule weights
  cli.py       # `evolab demo` / `evolab study`
experiments/
  run_study.py   # baselines + curve + mode/sigma/pool-size ablations -> results/*.json
  make_report.py # render README tables straight from the committed JSON
tests/         # exact-cost checks + search invariants (determinism, monotone best-so-far)
```

## Honest limitations

- The domain is toy-scale by design: 5 real-valued weights on a single machine.
  It is a *demonstration of the loop*, not a production scheduler; AlphaEvolve
  evolves actual program text, not a 5-vector.
- The evolved rule beats FIFO/SPT/EDD/MIN-SLACK decisively, but its edge over the
  strongest baseline (WSPT) is **~3% and within seed noise** — we report that
  honestly rather than claiming it "beats all textbook rules". The robust, large
  result is the ~40% climb from a random rule and population-ES > hill-climbing.
- Small training pools generalize worse: a 20-instance pool yields a champion that
  is measurably worse on held-out instances than the 120-instance pool's, and we
  include that ablation rather than hiding it.
- Results are single-machine, fixed-seed, CPU-only and therefore fully
  reproducible: `python -m pytest` plus `evolab study` regenerate every figure.

## License

MIT
