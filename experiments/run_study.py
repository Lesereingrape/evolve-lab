"""Full self-optimization study -> results/evolution.json (CPU, stdlib only).

Parts:
1. Textbook baselines scored on the same held-out pools the evolved rule uses.
2. Main population-ES curve across seeds (mean +/- std), plus the evolved weights.
3. Mode ablation: population-ES vs greedy (1+1) hill-climbing.
4. Mutation-strength (sigma) ablation.
5. Overfitting probe: shrink the training pool and watch the train-vs-held-out gap.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

from evolab.evolve import evolve
from evolab.policy import BASELINES, evaluate_rule, gen_jobs

N_JOBS = 30
GEN = 150


def pools(seed: int, n_train: int = 120, n_test: int = 200):
    rng = random.Random(seed)
    train = [gen_jobs(N_JOBS, rng) for _ in range(n_train)]
    test = [gen_jobs(N_JOBS, rng) for _ in range(n_test)]
    return train, test


def _mean_curve(runs, key):
    length = min(len(r["curve"]) for r in runs)
    return [{"generation": i,
             "mean": round(statistics.fmean(r["curve"][i][key] for r in runs), 3),
             "std": round(statistics.pstdev([r["curve"][i][key] for r in runs]), 3)
             if len(runs) > 1 else 0.0}
            for i in range(length)]


def run_study(seeds=(0, 1, 2), out="results/evolution.json"):
    t0 = time.time()

    # baselines on each seed's test pool
    base_acc: dict[str, list[float]] = {k: [] for k in BASELINES}
    for s in seeds:
        _, test = pools(s)
        for k, rule in BASELINES.items():
            base_acc[k].append(evaluate_rule(rule, test))
    baselines = {k: round(statistics.fmean(v), 2) for k, v in base_acc.items()}

    print("== main population-ES ==")
    main_runs = []
    evolved_w = []
    evolved_test = []
    for s in seeds:
        train, test = pools(s)
        r = evolve(train=train, probe=test, generations=GEN, mode="population", seed=s)
        main_runs.append(r)
        evolved_w.append(r["best"])
        evolved_test.append(r["curve"][-1]["probe"])
        print(f"  seed {s}: train {r['curve'][0]['train']:.0f} -> "
              f"{r['curve'][-1]['train']:.0f} | held-out {r['curve'][-1]['probe']:.0f}")

    print("== mode ablation ==")
    hc_runs = []
    for s in seeds:
        train, test = pools(s)
        hc_runs.append(evolve(train=train, probe=test, generations=GEN,
                              mode="hill_climb", seed=s))

    print("== sigma ablation ==")
    sigmas = {}
    for sg in (0.1, 0.4, 0.8):
        rr = []
        for s in seeds[:2]:
            train, test = pools(s)
            rr.append(evolve(train=train, probe=test, generations=GEN, sigma=sg,
                             mode="population", seed=s))
        sigmas[str(sg)] = round(statistics.fmean(x["curve"][-1]["probe"] for x in rr), 2)

    print("== train-pool size (overfitting probe) ==")
    pool_sizes = {}
    for nt in (20, 120):
        tr_gaps, pv = [], []
        for s in seeds[:2]:
            train, test = pools(s, n_train=nt)
            r = evolve(train=train, probe=test, generations=GEN, mode="population", seed=s)
            tr_gaps.append(r["curve"][-1]["train"])
            pv.append(r["curve"][-1]["probe"])
        pool_sizes[str(nt)] = {"train": round(statistics.fmean(tr_gaps), 2),
                               "held_out": round(statistics.fmean(pv), 2)}

    best_mean_w = [round(statistics.fmean(w[i] for w in evolved_w), 3)
                   for i in range(len(evolved_w[0]))]
    curve = _mean_curve(main_runs, "train")
    probe_curve = _mean_curve(main_runs, "probe")
    evolved_final = round(statistics.fmean(evolved_test), 2)
    best_baseline = min(baselines.values())

    result = {
        "config": {"n_jobs": N_JOBS, "generations": GEN, "seeds": list(seeds),
                   "n_train": 120, "n_test": 200},
        "baselines_heldout": baselines,
        "main": {
            "curve_train": curve,
            "curve_heldout": probe_curve,
            "start": curve[0]["mean"],
            "final_train": curve[-1]["mean"],
            "final_heldout": evolved_final,
            "reduction_vs_start_pct": round(100 * (curve[0]["mean"] - evolved_final)
                                            / curve[0]["mean"], 1),
            "gain_vs_best_baseline_pct": round(100 * (best_baseline - evolved_final)
                                               / best_baseline, 1),
            "evolved_weights_mean": dict(zip(
                ["inv_proc", "weight", "slack", "wratio", "arrival"], best_mean_w, strict=True)),
        },
        "ablation_mode": {
            "population_heldout": evolved_final,
            "hill_climb_heldout": round(statistics.fmean(
                r["curve"][-1]["probe"] for r in hc_runs), 2),
            "hill_climb_curve": _mean_curve(hc_runs, "train"),
        },
        "ablation_sigma_heldout": sigmas,
        "ablation_train_pool_size": pool_sizes,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(out)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"\nevolved held-out {evolved_final} vs best baseline {best_baseline} "
          f"({result['main']['gain_vs_best_baseline_pct']}% better)")
    print(f"wrote {path} in {result['runtime_sec']}s")


if __name__ == "__main__":
    run_study()
