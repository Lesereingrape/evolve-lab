"""Guard the README's hand-written figures against the committed artifact.

The results block is byte-pinned to ``results/evolution.json`` by
``test_readme_matches_results.py``; the claims in the prose *around* that block ("~40%
climb", "~3% edge over WSPT", "5 real-valued weights", the pool sizes, the wall-clock
budgets) are typed by hand, so each one is re-derived here from the artifact or the code.
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


def test_readme_names_the_std_convention_the_tables_use():
    """`+/-` is ambiguous unless the file says which divisor produced it.

    The published spreads are the population standard deviation over seeds, so the
    README has to use that word: a reader who recomputed the other convention would
    land on a different number and conclude the tables were wrong.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search("population[^.]{0,60}standard\\s+deviation", readme), (
        "the README no longer states which standard-deviation convention its "
        "`+/-` columns use")


def test_the_published_wall_clock_is_the_one_the_artifact_records():
    """The rerun note pairs two runtimes and only the committed half of that is checkable."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    named = re.findall(r"published\s+(\d+(?:\.\d+)?)s(?![\d])", readme)
    assert named, "the rerun note no longer names the published runtime"
    assert len(named) == 1, f"the README names the published runtime more than once: {named}"
    artifact = json.loads((ROOT / "results" / "evolution.json").read_text(encoding="utf-8"))
    assert float(named[0]) == artifact["runtime_sec"], (
        f"README says the published run took {named[0]}s, "
        f"results/evolution.json records {artifact['runtime_sec']}s")


def test_the_promised_study_length_is_the_length_that_was_measured():
    """The blurb and the limitations section both budget the study in minutes.

    Two sentences make the same promise, so they are checked as a set against one
    artifact field: a study that grows past its advertised budget should have to move
    both mentions, not quietly outlive one of them.
    """
    budgets = {int(m.group(1)) for m in re.finditer(r"~(\d+) minutes", README)}
    assert budgets, "the README no longer budgets the full study; drop this guard with it"
    minutes = DATA["runtime_sec"] / 60.0
    for budget in budgets:
        assert 0.5 * budget <= minutes <= 2.0 * budget, (
            f"the README calls the study a ~{budget}-minute job; the committed run took "
            f"{minutes:.1f} minutes")


def test_the_documented_rerun_writes_a_relative_scratch_file():
    """`--out /tmp/...` is not one path across shells, so the recipe must not use it."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--out /tmp/" not in readme, (
        "the rerun recipe is back to a /tmp path; Git-Bash rewrites it before the script "
        "sees it, so use a relative scratch file")
    assert "again-check.json" in readme, (
        "the rerun recipe no longer names the relative scratch file it documents")
