"""Tests for the evolutionary search properties."""

from __future__ import annotations

import random

from evolab.evolve import crossover, evolve, mutate, random_genome
from evolab.policy import Weights, evaluate_rule, gen_jobs


def _pool(seed, n_inst=40, n_jobs=25):
    rng = random.Random(seed)
    return [gen_jobs(n_jobs, rng) for _ in range(n_inst)]


def test_genome_ops_preserve_length():
    rng = random.Random(0)
    g = random_genome(rng)
    assert len(g) == 5
    assert len(mutate(g, rng, 0.3)) == 5
    assert len(crossover(g, random_genome(rng), rng)) == 5


def test_crossover_only_takes_from_parents():
    rng = random.Random(1)
    a = [1.0] * 5
    b = [2.0] * 5
    child = crossover(a, b, rng)
    assert all(x in (1.0, 2.0) for x in child)


def test_evolution_reduces_cost_and_is_deterministic():
    train = _pool(5)
    r1 = evolve(train=train, generations=40, mode="population", seed=7)
    r2 = evolve(train=train, generations=40, mode="population", seed=7)
    assert r1["best"] == r2["best"]
    rule = Weights.from_vec(r1["best"]).as_rule()
    # champion is never worse than the generation-0 best (best-so-far is monotone)
    assert evaluate_rule(rule, train) <= r1["curve"][0]["train"] + 1e-6


def test_best_so_far_is_monotone_nonincreasing():
    train = _pool(9)
    r = evolve(train=train, generations=30, mode="hill_climb", seed=2)
    vals = [pt["train"] for pt in r["curve"]]
    assert all(vals[i + 1] <= vals[i] + 1e-9 for i in range(len(vals) - 1))
