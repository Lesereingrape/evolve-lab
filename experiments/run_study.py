"""Full self-optimization study -> results/evolution.json (CPU, stdlib only).

    python experiments/run_study.py [--out PATH]

Parts:
1. Textbook baselines scored on the same held-out pools the evolved rule uses.
2. Main population-ES curve across seeds (mean +/- std), plus the evolved weights.
3. Mode ablation: population-ES vs greedy (1+1) hill-climbing.
4. Mutation-strength (sigma) ablation.
5. Overfitting probe: shrink the training pool and watch the train-vs-held-out gap.

Every aggregate written here is reduced from the per-seed search traces committed
alongside it, so ``tests/test_artifact_is_internally_consistent.py`` can recompute the
published tables from those raw traces instead of taking them on trust.

``--out`` exists so a verification rerun can be written to a scratch path and diffed
field for field against the committed artifact without overwriting it.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from evolab.evolve import evolve
from evolab.policy import BASELINES, evaluate_rule, gen_jobs
from evolab.provenance import environment

N_JOBS = 30
GEN = 150
SEEDS = (0, 1, 2)
SIGMAS = (0.1, 0.4, 0.8)
POOL_SIZES = (20, 120)
WEIGHT_NAMES = ("inv_proc", "weight", "slack", "wratio", "arrival")


def pools(seed: int, n_train: int = 120, n_test: int = 200):
    rng = random.Random(seed)
    train = [gen_jobs(N_JOBS, rng) for _ in range(n_train)]
    test = [gen_jobs(N_JOBS, rng) for _ in range(n_test)]
    return train, test


def _mean_curve(curves: list[list[float]]) -> list[dict]:
    """Best-so-far mean +/- population std per generation, over raw per-seed curves."""
    length = min(len(c) for c in curves)
    return [{"generation": i,
             "mean": round(statistics.fmean(c[i] for c in curves), 3),
             "std": round(statistics.pstdev([c[i] for c in curves]), 3)
             if len(curves) > 1 else 0.0}
            for i in range(length)]


def _mean(vals: list[float]) -> float:
    return round(statistics.fmean(vals), 2)


def run_study(seeds=SEEDS, generations=GEN, out="results/evolution.json"):
    t0 = time.time()
    per: dict[str, dict] = {}
    # The sigma and pool-size sweeps re-run the whole search per setting, so they use
    # a subset; the artifact records which seeds, and their aggregates average those.
    ablation_seeds = seeds[:2]

    for s in seeds:
        print(f"== seed {s} ==")
        train, test = pools(s)
        row: dict = {
            "baselines_heldout": {k: evaluate_rule(rule, test) for k, rule in BASELINES.items()},
        }
        pop = evolve(train=train, probe=test, generations=generations, mode="population", seed=s)
        row["main_curve_train"] = [p["train"] for p in pop["curve"]]
        row["main_curve_probe"] = [p["probe"] for p in pop["curve"]]
        row["best_weights"] = pop["best"]
        print(f"  population: train {pop['curve'][0]['train']:.0f} -> "
              f"{pop['curve'][-1]['train']:.0f} | held-out {pop['curve'][-1]['probe']:.0f}")
        hc = evolve(train=train, probe=test, generations=generations, mode="hill_climb", seed=s)
        row["hill_climb_curve_train"] = [p["train"] for p in hc["curve"]]
        row["hill_climb_heldout"] = hc["curve"][-1]["probe"]
        print(f"  hill_climb: held-out {row['hill_climb_heldout']:.0f}")
        if s in ablation_seeds:
            row["sigma_heldout"] = {}
            for sg in SIGMAS:
                rr = evolve(train=train, probe=test, generations=generations, sigma=sg,
                            mode="population", seed=s)
                row["sigma_heldout"][str(sg)] = rr["curve"][-1]["probe"]
            row["pool_size"] = {}
            for nt in POOL_SIZES:
                small_train, small_test = pools(s, n_train=nt)
                pr = evolve(train=small_train, probe=small_test, generations=generations,
                            mode="population", seed=s)
                row["pool_size"][str(nt)] = {"train": pr["curve"][-1]["train"],
                                             "held_out": pr["curve"][-1]["probe"]}
        per[str(s)] = row

    rows = [per[str(s)] for s in seeds]
    arows = [per[str(s)] for s in ablation_seeds]

    baselines = {k: _mean([r["baselines_heldout"][k] for r in rows]) for k in BASELINES}
    curve = _mean_curve([r["main_curve_train"] for r in rows])
    probe_curve = _mean_curve([r["main_curve_probe"] for r in rows])
    evolved_final = _mean([r["main_curve_probe"][-1] for r in rows])
    best_baseline = min(baselines.values())
    best_mean_w = [round(statistics.fmean(r["best_weights"][i] for r in rows), 3)
                   for i in range(len(WEIGHT_NAMES))]

    result = {
        "config": {"n_jobs": N_JOBS, "generations": generations,
                   "seeds": list(seeds), "ablation_seeds": list(ablation_seeds),
                   "n_train": 120, "n_test": 200},
        "environment": environment(),
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
            "evolved_weights_mean": dict(zip(WEIGHT_NAMES, best_mean_w, strict=True)),
        },
        "ablation_mode": {
            "population_heldout": evolved_final,
            "hill_climb_heldout": _mean([r["hill_climb_heldout"] for r in rows]),
            "hill_climb_curve": _mean_curve([r["hill_climb_curve_train"] for r in rows]),
        },
        "ablation_sigma_heldout": {
            str(sg): _mean([r["sigma_heldout"][str(sg)] for r in arows]) for sg in SIGMAS},
        "ablation_train_pool_size": {
            str(nt): {"train": _mean([r["pool_size"][str(nt)]["train"] for r in arows]),
                      "held_out": _mean([r["pool_size"][str(nt)]["held_out"] for r in arows])}
            for nt in POOL_SIZES},
        # The raw search traces every aggregate above is reduced from.
        "per_seed": per,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(out)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nevolved held-out {evolved_final} vs best baseline {best_baseline} "
          f"({result['main']['gain_vs_best_baseline_pct']}% better)")
    print(f"wrote {path} in {result['runtime_sec']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_study")
    parser.add_argument("--out", default="results/evolution.json",
                        help="where to write the artifact; point it at a scratch path "
                             "to rerun and diff against the committed one")
    args = parser.parse_args()
    run_study(out=args.out)
