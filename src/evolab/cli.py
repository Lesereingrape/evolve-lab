"""Command line entry point: ``evolab``.

    evolab demo    # one seeded evolution run, prints start -> held-out
    evolab study   # full multi-seed study -> results/evolution.json
"""

from __future__ import annotations

import argparse
import random

from .evolve import evolve
from .policy import BASELINES, evaluate_rule, gen_jobs


def _pools(seed: int, n_train: int = 120, n_test: int = 200, n_jobs: int = 30):
    rng = random.Random(seed)
    train = [gen_jobs(n_jobs, rng) for _ in range(n_train)]
    test = [gen_jobs(n_jobs, rng) for _ in range(n_test)]
    return train, test


def _demo(seed: int, generations: int) -> None:
    train, test = _pools(seed)
    print("held-out baseline (mean total weighted tardiness, lower is better):")
    for k, rule in sorted(BASELINES.items(), key=lambda kv: evaluate_rule(kv[1], test)):
        print(f"  {k:9s} {evaluate_rule(rule, test):8.1f}")
    r = evolve(train=train, probe=test, generations=generations, mode="population",
               seed=seed)
    c = r["curve"]
    print(f"\npopulation-ES: train {c[0]['train']:.0f} -> {c[-1]['train']:.0f}, "
          f"held-out {c[-1]['probe']:.0f}")
    names = ["inv_proc", "weight", "slack", "wratio", "arrival"]
    weights = dict(zip(names, [round(x, 2) for x in r["best"]], strict=True))
    print(f"evolved weights: {weights}")
    print("interpretation: a positive wratio term means it rediscovered WSPT; the"
          "\nother non-zero terms are the learned blend on top of it.")


def _study(generations: int) -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
    from run_study import run_study

    run_study()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evolab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="single seeded evolution run")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--generations", type=int, default=150)
    s = sub.add_parser("study", help="full study -> results/evolution.json")
    s.add_argument("--generations", type=int, default=150)
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        _demo(args.seed, args.generations)
    else:
        _study(args.generations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
