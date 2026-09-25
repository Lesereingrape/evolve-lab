"""The published evolution tables must be recomputable from the committed traces.

``results/evolution.json`` stores the raw per-seed best-so-far curves next to the
aggregates rendered into the README. This test re-derives every aggregate from those
traces with the same reduction the study used, so a published number that does not
follow from the recorded search cannot survive CI.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "evolution.json").read_text(encoding="utf-8"))
PER = DATA["per_seed"]
SEEDS = DATA["config"]["seeds"]
ABLATION_SEEDS = DATA["config"]["ablation_seeds"]
ROWS = [PER[str(s)] for s in SEEDS]
AROWS = [PER[str(s)] for s in ABLATION_SEEDS]


def _mean_curve(curves):
    length = min(len(c) for c in curves)
    return [{"generation": i,
             "mean": round(statistics.fmean(c[i] for c in curves), 3),
             "std": round(statistics.pstdev([c[i] for c in curves]), 3)
             if len(curves) > 1 else 0.0}
            for i in range(length)]


def _mean(vals):
    return round(statistics.fmean(vals), 2)


def test_artifact_covers_every_seed_and_the_sweeps_record_theirs():
    assert sorted(PER) == sorted(str(s) for s in SEEDS)
    assert all(s in SEEDS for s in ABLATION_SEEDS)
    for s in SEEDS:
        row = PER[str(s)]
        assert len(row["main_curve_train"]) == DATA["config"]["generations"] + 1
        assert len(row["main_curve_probe"]) == len(row["main_curve_train"])
        assert len(row["hill_climb_curve_train"]) == len(row["main_curve_train"])
        assert len(row["best_weights"]) == len(DATA["main"]["evolved_weights_mean"])
        swept = "sigma_heldout" in row and "pool_size" in row
        assert swept == (s in ABLATION_SEEDS), s


def test_baselines_are_the_seed_mean_of_the_recorded_scores():
    for rule, published in DATA["baselines_heldout"].items():
        assert published == _mean([r["baselines_heldout"][rule] for r in ROWS]), rule


def test_main_curves_are_recomputed_from_the_raw_traces():
    assert DATA["main"]["curve_train"] == _mean_curve([r["main_curve_train"] for r in ROWS])
    assert DATA["main"]["curve_heldout"] == _mean_curve([r["main_curve_probe"] for r in ROWS])


def test_headline_numbers_follow_from_the_recomputed_curves():
    main = DATA["main"]
    assert main["start"] == main["curve_train"][0]["mean"]
    assert main["final_train"] == main["curve_train"][-1]["mean"]
    assert main["final_heldout"] == _mean([r["main_curve_probe"][-1] for r in ROWS])
    assert main["reduction_vs_start_pct"] == round(
        100 * (main["start"] - main["final_heldout"]) / main["start"], 1)
    best = min(DATA["baselines_heldout"].values())
    assert main["gain_vs_best_baseline_pct"] == round(
        100 * (best - main["final_heldout"]) / best, 1)
    assert main["evolved_weights_mean"] == dict(zip(
        ["inv_proc", "weight", "slack", "wratio", "arrival"],
        [round(statistics.fmean(r["best_weights"][i] for r in ROWS), 3)
         for i in range(5)], strict=True))


def test_mode_and_sigma_and_pool_ablations_match_their_traces():
    ab = DATA["ablation_mode"]
    assert ab["population_heldout"] == DATA["main"]["final_heldout"]
    assert ab["hill_climb_heldout"] == _mean([r["hill_climb_heldout"] for r in ROWS])
    assert ab["hill_climb_curve"] == _mean_curve(
        [r["hill_climb_curve_train"] for r in ROWS])
    for sg, published in DATA["ablation_sigma_heldout"].items():
        assert published == _mean([r["sigma_heldout"][sg] for r in AROWS]), sg
    for nt, published in DATA["ablation_train_pool_size"].items():
        assert published == {
            "train": _mean([r["pool_size"][nt]["train"] for r in AROWS]),
            "held_out": _mean([r["pool_size"][nt]["held_out"] for r in AROWS])}, nt


def test_tardiness_is_a_real_cost_and_heldout_tracks_train():
    for row in ROWS:
        assert row["main_curve_train"][-1] <= row["main_curve_train"][0]
        assert row["main_curve_probe"][-1] > 0
    # The study claims the champion does not over-fit its training pool, which is only
    # true while held-out cost stays close to train cost; this is that claim as a number.
    rel_gap = ((DATA["main"]["final_heldout"] - DATA["main"]["final_train"])
               / DATA["main"]["final_train"])
    assert rel_gap < 0.05, rel_gap


def test_environment_records_the_interpreter_behind_the_traces():
    env = DATA["environment"]
    for key in ("python", "platform", "device"):
        assert isinstance(env[key], str) and env[key], key
