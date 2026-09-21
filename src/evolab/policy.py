"""Single-machine dispatching: an exact simulator, textbook rules, and a learned rule.

The agent schedules jobs on one machine, no preemption, and is scored by **total
weighted tardiness**  sum of w * max(0, completion - due)  - a classic NP-hard
objective. The cost is computed *exactly* by ``simulate``, so it is a true
verifier, not a learned proxy: that is the whole reason a tiny evolutionary loop
can honestly self-improve on a CPU in seconds.

A dispatching *rule* picks which released job to run next. We compare against the
textbook non-delay rules and against a parametric rule whose five weights are the
evolved genome:

    priority(job) = a*(1/p) + b*w + c*(slack) + d*(w/p) + e*(arrival)

with slack = -(due - p) and arrival = -release. Each textbook rule is roughly a
one-hot setting of this vector (SPT->a, weight->b, EDD->c, WSPT->d, FIFO->e), so
the search space *contains* the known-good rules - a good heuristic search should
rediscover and blend them without ever being shown one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# A job: release time, processing time, weight (tardiness cost), due date.
Job = tuple[float, float, float, float]


def gen_jobs(n: int, rng: random.Random,
             p_lo: float = 1.0, p_hi: float = 10.0,
             w_lo: float = 1.0, w_hi: float = 10.0,
             due_lo: float = 1.5, due_hi: float = 6.0) -> list[Job]:
    jobs: list[Job] = []
    clock = 0.0
    for _ in range(n):
        release = clock
        p = rng.uniform(p_lo, p_hi)
        w = rng.uniform(w_lo, w_hi)
        clock += p * rng.uniform(0.3, 0.8)
        due = release + p * rng.uniform(due_lo, due_hi)
        jobs.append((release, p, w, due))
    return jobs


def simulate(jobs: list[Job], priority) -> float:
    """Non-delay dispatch under ``priority(job, now)`` (higher runs first).

    Returns total weighted tardiness. Deterministic and exact.
    """
    unscheduled = list(jobs)
    now = 0.0
    total = 0.0
    while unscheduled:
        available = [j for j in unscheduled if j[0] <= now + 1e-9]
        if not available:
            now = min(j[0] for j in unscheduled)
            continue
        nxt = max(available, key=lambda j: priority(j, now))
        unscheduled.remove(nxt)
        now += nxt[1]
        total += nxt[2] * max(0.0, now - nxt[3])
    return total


# Textbook baselines -----------------------------------------------------------
def FIFO(j: Job, now: float) -> float:
    return -j[0]


def SPT(j: Job, now: float) -> float:
    return -j[1]


def EDD(j: Job, now: float) -> float:
    return -j[3]


def WSPT(j: Job, now: float) -> float:
    return j[2] / j[1]


def MINSLACK(j: Job, now: float) -> float:
    return -(j[3] - j[1])


BASELINES = {"FIFO": FIFO, "SPT": SPT, "EDD": EDD, "WSPT": WSPT, "MINSLACK": MINSLACK}


# Learned parametric rule ------------------------------------------------------
@dataclass(frozen=True)
class Weights:
    inv_proc: float
    weight: float
    slack: float
    wratio: float
    arrival: float

    @staticmethod
    def from_vec(v: list[float]) -> Weights:
        return Weights(*v)

    def as_rule(self):
        w = self

        def rule(j: Job, now: float) -> float:
            _release, p, weight, due = j
            return (w.inv_proc / p
                    + w.weight * weight
                    - w.slack * (due - p)
                    + w.wratio * (weight / p)
                    - w.arrival * j[0])
        return rule


def evaluate_rule(rule, instances: list[list[Job]]) -> float:
    return sum(simulate(inst, rule) for inst in instances) / len(instances)
