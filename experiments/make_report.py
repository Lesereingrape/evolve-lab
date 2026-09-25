"""Render README result tables directly from results/evolution.json.

    python experiments/make_report.py [--write] [--results PATH]

``--write`` splices the rendered block between the ``RESULTS:START``/``RESULTS:END``
markers in README.md, so the published prose is generated from the artifact instead of
retyped by hand; ``tests/test_readme_matches_results.py`` fails if the two drift.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _sample(curve, step=15):
    idx = list(range(0, len(curve), step))
    if idx[-1] != len(curve) - 1:
        idx.append(len(curve) - 1)
    return [curve[i] for i in idx]


def build(data: dict) -> str:
    cfg = data["config"]
    main = data["main"]
    out: list[str] = []
    seeds = ", ".join(str(s) for s in cfg["seeds"])
    out.append("*Every figure below is produced by `experiments/run_study.py` on CPU "
               "and stored in the committed "
               "[`results/evolution.json`](results/evolution.json); the tables are "
               f"rendered by `experiments/make_report.py`. {len(cfg['seeds'])} seeds "
               f"({seeds}), {cfg['generations']} generations, {cfg['n_train']} "
               f"training / {cfg['n_test']} held-out instances of {cfg['n_jobs']} "
               "jobs each.*")
    out.append("")
    out.append(f"- objective: total weighted tardiness on {cfg['n_jobs']}-job "
               f"single-machine instances (lower is better)")
    env = data["environment"]
    out.append(f"- measured under: Python {env['python']} on {env['platform']}, "
               f"{env['device']} — the search is a seeded pure-Python computation, so "
               "`experiments/run_study.py --out again-check.json` reruns it exactly and "
               "`make_report.py --write` re-renders these tables; only `runtime_sec` "
               "is allowed to differ")
    out.append("- every mean, std and curve point below is reduced from the raw "
               "per-seed traces stored under `per_seed`, and "
               "`tests/test_artifact_is_internally_consistent.py` recomputes them from "
               "those traces")
    out.append("")

    out.append("### Held-out cost: evolved rule vs textbook dispatching rules\n")
    out.append("| rule | held-out tardiness | vs evolved |")
    out.append("|------|-------------------:|---------:|")
    ev = main["final_heldout"]
    ev_std = main["curve_heldout"][-1]["std"]
    for name, val in sorted(data["baselines_heldout"].items(), key=lambda kv: kv[1]):
        rel = f"{100 * (val - ev) / val:+.1f}%"
        out.append(f"| {name} | {val:.1f} | {rel} |")
    out.append(f"| **Evolved (population-ES)** | **{ev:.1f}** | — |")
    out.append("")
    out.append("`vs evolved` is how much worse each baseline is than the discovered "
               "rule (positive = the evolved rule is better). The evolved value is a "
               f"mean over seeds (**±{ev_std:.0f}**), so a small edge over the "
               "strongest baseline (WSPT) is within seed noise.")
    out.append("")

    out.append("### Self-improvement curve (best-so-far, mean over seeds)\n")
    out.append("| generation | train cost | held-out cost |")
    out.append("|-----------:|-----------:|--------------:|")
    tc = _sample(main["curve_train"])
    hc = _sample(main["curve_heldout"])
    for a, b in zip(tc, hc, strict=True):
        out.append(f"| {a['generation']} | {a['mean']:.0f} | {b['mean']:.0f} |")
    out.append("")
    out.append(f"From a **random** rule ({main['start']:.0f}) to {ev:.0f} "
               f"(±{ev_std:.0f}): a **{main['reduction_vs_start_pct']}%** reduction. "
               f"The self-improvement is large and unambiguous against the *start* "
               f"and against FIFO/SPT/EDD/MIN-SLACK; the extra "
               f"{main['gain_vs_best_baseline_pct']}% over WSPT is a small edge that "
               f"sits near the seed-to-seed spread.")
    out.append("")

    out.append("### Ablations\n")
    out.append("Search mode (held-out cost):")
    ab = data["ablation_mode"]
    out.append(f"- population-ES: **{ab['population_heldout']:.0f}**")
    out.append(f"- greedy (1+1) hill-climb: {ab['hill_climb_heldout']:.0f}")
    out.append("")
    n_abl = len(cfg["ablation_seeds"])
    out.append(f"Mutation strength sigma (held-out cost, {n_abl} seeds):")
    for s, v in sorted(data["ablation_sigma_heldout"].items(), key=lambda kv: float(kv[0])):
        out.append(f"- sigma={s}: {v:.0f}")
    out.append("")
    out.append(f"Training-pool size → generalization (held-out cost, {n_abl} seeds). The "
               "search only ever sees the training pool, so held-out cost is the "
               "honest read; the small pool over-fits its few instances and transfers "
               "worse.")
    for nt, v in sorted(data["ablation_train_pool_size"].items(), key=lambda kv: int(kv[0])):
        out.append(f"- pool={nt}: held-out **{v['held_out']:.0f}** "
                   f"(train on that pool's own instances: {v['train']:.0f})")
    out.append("")

    w = main["evolved_weights_mean"]
    out.append("### What the agent discovered\n")
    out.append("Mean evolved weights: "
               + ", ".join(f"`{k}={v}`" for k, v in w.items())
               + ". A dominant positive `wratio` term with a secondary `inv_proc` "
               "term means the search **rediscovered WSPT and blended in some "
               "shortest-processing-time pressure from scratch**, never being shown "
               "either rule. That blend decisively beats FIFO/SPT/EDD/MIN-SLACK and "
               "*matches or very slightly edges* the WSPT baseline — the honest "
               f"headline is the {main['reduction_vs_start_pct']}% climb from a random "
               "rule, not a big lead over the best textbook rule.")
    return "\n".join(out)


def _write(rendered: str, readme: Path) -> None:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r"\n.*?\n" + re.escape(END), re.DOTALL)
    assert pattern.search(text), f"README is missing the {START}/{END} markers"
    readme.write_text(pattern.sub(lambda _: f"{START}\n{rendered}\n{END}", text),
                      encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="make_report")
    ap.add_argument("--results", default=str(ROOT / "results" / "evolution.json"))
    ap.add_argument("--write", action="store_true",
                    help="splice the rendered block into README.md instead of printing")
    args = ap.parse_args()
    block = build(json.loads(Path(args.results).read_text(encoding="utf-8")))
    if args.write:
        _write(block, ROOT / "README.md")
        print("README results block rewritten")
    else:
        print(block)
