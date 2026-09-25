"""Guard the README's hand-written figures against the committed artifact.

The results block is byte-pinned to ``results/evolution.json`` by
``test_readme_matches_results.py``; the claims in the prose *around* that block ("~40%
climb", "~3% edge over WSPT", "5 real-valued weights", the pool sizes) are typed by
hand, so each one is re-derived here from the artifact or the code instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DATA = json.loads((ROOT / "results" / "evolution.json").read_text(encoding="utf-8"))
MAIN = DATA["main"]


def test_headline_reduction_claim_matches_the_artifact():
    claims = [float(m) for m in re.findall(r"~(\d+)% climb from a random rule", README)]
    assert claims, "the '~N% climb from a random rule' sentence was reworded"
    for claim in claims:
        assert abs(claim - MAIN["reduction_vs_start_pct"]) <= 1.0, (
            f"README says a ~{claim}% climb; the artifact measured "
            f"{MAIN['reduction_vs_start_pct']}%")


def test_wspt_edge_claim_matches_the_artifact():
    m = re.search(r"strongest baseline \(WSPT\) is \*\*~(\d+(?:\.\d+)?)%", README)
    assert m, "the WSPT-edge sentence was reworded; update this test with it"
    claim = float(m.group(1))
    assert abs(claim - MAIN["gain_vs_best_baseline_pct"]) <= 1.0, (
        f"README claims a ~{claim}% edge over WSPT; the artifact says "
        f"{MAIN['gain_vs_best_baseline_pct']}%")


def test_genome_size_claim_matches_the_evolved_weights():
    m = re.search(r"(\d+) real-valued weights", README)
    assert m, "the genome-size claim was reworded"
    assert int(m.group(1)) == len(MAIN["evolved_weights_mean"]), (
        f"README says {m.group(1)} weights; the evolved genome has "
        f"{len(MAIN['evolved_weights_mean'])}")


def test_pool_size_and_seed_claims_exist_in_the_artifact():
    cfg = DATA["config"]
    for nt in DATA["ablation_train_pool_size"]:
        assert f"{nt}-instance pool" in README or f"pool={nt}" in README, nt
    assert f"{cfg['n_train']} training" in README or "120 training" in README
    n_abl = len(cfg["ablation_seeds"])
    assert f"{n_abl} seeds" in README, (
        f"the sweep captions should state the {n_abl} ablation seeds the artifact "
        "actually averages")
    assert f"{len(cfg['seeds'])} seeds" in README


def test_generations_claim_matches_the_recorded_curves():
    m = re.search(r"(\d+) generations", README)
    assert m, "the generations claim was reworded"
    assert int(m.group(1)) == DATA["config"]["generations"]
    longest = max(len(r["main_curve_train"]) for r in DATA["per_seed"].values())
    assert longest == DATA["config"]["generations"] + 1, (
        "every seed's curve should span generation 0 through the last generation")
