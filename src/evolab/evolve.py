"""A minimal, dependency-free evolutionary search that self-optimizes the rule.

The small-scale analog of AlphaEvolve / the Darwin-Gödel Machine: keep a
population of candidate genomes (dispatch-rule weights), mutate and recombine them,
and keep whatever the *exact* tardiness simulator scores best. "Self-improving"
here means the loop measurably lowers total weighted tardiness from simulation
feedback alone — no learned reward model, no API, no rule hand-written by us.

Two search modes share one interface so they can be compared head-to-head:
  * population  - a (mu, lambda)-ES with elitism and uniform crossover.
  * hill_climb  - a (1+1)-EA: one parent, one mutant per step, keep the better.
"""

from __future__ import annotations

import random

from .policy import Weights, evaluate_rule

GENOME_LEN = 5


def _fit(vec: list[float], instances: list[list]) -> float:
    return evaluate_rule(Weights.from_vec(vec).as_rule(), instances)


def random_genome(rng: random.Random, scale: float = 1.0) -> list[float]:
    return [rng.uniform(-scale, scale) for _ in range(GENOME_LEN)]


def mutate(vec: list[float], rng: random.Random, sigma: float) -> list[float]:
    return [x + rng.gauss(0.0, sigma) for x in vec]


def crossover(a: list[float], b: list[float], rng: random.Random) -> list[float]:
    return [x if rng.random() < 0.5 else y for x, y in zip(a, b, strict=True)]


def evolve(
    *,
    train: list[list],
    probe: list[list] | None = None,
    generations: int = 150,
    pop: int = 24,
    lambda_: int | None = None,
    sigma: float = 0.4,
    mode: str = "population",
    seed: int = 0,
) -> dict:
    """Run evolution; return the best-so-far curve (train, and probe if given) + winner.

    Recording a *probe* set the search never optimizes against is deliberate: if
    the champion overfits the training pool, train and probe curves diverge — a
    property this repo surfaces rather than hides.
    """
    rng = random.Random(seed)
    lambda_ = lambda_ or pop

    def snap(vec: list[float], gen: int) -> dict:
        point = {"generation": gen, "train": round(_fit(vec, train), 3)}
        if probe is not None:
            point["probe"] = round(_fit(vec, probe), 3)
        return point

    if mode == "hill_climb":
        parent = random_genome(rng)
        pfit = _fit(parent, train)
        curve = [snap(parent, 0)]
        for g in range(1, generations + 1):
            child = mutate(parent, rng, sigma)
            cfit = _fit(child, train)
            if cfit <= pfit:
                parent, pfit = child, cfit
            curve.append(snap(parent, g))
        return {"curve": curve, "best": parent, "best_train": pfit, "mode": mode}

    population = sorted((random_genome(rng) for _ in range(pop)),
                        key=lambda v: _fit(v, train))
    best_vec = population[0]
    best_val = _fit(best_vec, train)
    curve = [snap(best_vec, 0)]
    for g in range(1, generations + 1):
        elites = population[: max(1, pop // 4)]
        offspring: list[list[float]] = []
        while len(offspring) < lambda_:
            child = mutate(crossover(rng.choice(elites), rng.choice(population), rng),
                           rng, sigma)
            offspring.append(child)
        population = sorted(elites + offspring, key=lambda v: _fit(v, train))[:pop]
        if _fit(population[0], train) < best_val:
            best_vec = population[0]
            best_val = _fit(best_vec, train)
        curve.append(snap(best_vec, g))
    return {"curve": curve, "best": best_vec, "best_train": best_val, "mode": mode}
