"""Phase 2: an adversarial debate between two XSL theories over the corpus.

Each round, the agent advocating each theory proposes a *corpus query*: a set
of conditions plus a statistic of their per-item accuracies (Python source)
on which its theory should match people better than the opponent. A query is
evaluated on the human data and on both theories' population-mean
predictions (`fitting.population_accuracy`, which averages over the fitted
between-subject ability spread). It counts only if the data can decide it:
the two predictions must differ by at least `SEPARATION` standard errors of
the human value, where the standard error comes from resampling people
within each condition. An arbiter then reads both outcomes and decides which
theory to revise, and how.

Mode B: the whole corpus is visible to the agents, so a query is a choice of
diagnostic contrast, not a blind prediction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import AliasChoices, Field, PrivateAttr

from src.promptable import Promptable
from src.xsl.corpus import Condition

SEPARATION = 2.0     # predictions must differ by this many human SEs
N_BOOTSTRAP = 500


class CorpusQuery(Promptable):
    conditions: list[str] = Field(
        ...,
        description="Labels of the corpus conditions the statistic uses (at least one).",
    )
    metric_source: str = Field(
        ...,
        validation_alias=AliasChoices("metric", "metric_source"),
        description=(
            "Python source defining def metric(acc) -> float, where acc maps each listed condition "
            "label to a numpy array of per-item accuracies (proportion correct, in corpus item order). "
            "`np` is available. Example: return acc['221'].mean() - acc['222'].mean()"
        ),
    )
    claim: str = Field(
        ...,
        description="Which theory should match people better on this statistic, and the mechanism that makes it so.",
    )

    _impl: Any = PrivateAttr(default=None)

    def model_post_init(self, _context: Any) -> None:
        if not self.conditions:
            raise ValueError("a query needs at least one condition")
        ns: dict[str, Any] = {"np": np}
        exec(self.metric_source, ns)
        if not callable(ns.get("metric")):
            raise ValueError("metric_source must define a function named `metric`")
        object.__setattr__(self, "_impl", ns["metric"])

    def value(self, acc: dict[str, np.ndarray]) -> float:
        v = float(self._impl({k: acc[k] for k in self.conditions}))
        if not math.isfinite(v):
            raise ValueError(f"metric returned {v}")
        return v


@dataclass
class QueryOutcome:
    proposer: str                       # label of the theory whose advocate proposed it
    query: CorpusQuery
    human: float
    human_se: float
    predictions: dict[str, float]       # theory label -> predicted statistic
    separable: bool
    winner: str | None                  # closer to the human value, if separable

    def errors_in_se(self) -> dict[str, float]:
        se = max(self.human_se, 1e-12)
        return {k: abs(v - self.human) / se for k, v in self.predictions.items()}

    def to_json(self) -> dict:
        return {"proposer": self.proposer, "query": self.query.model_dump(), "human": self.human,
                "human_se": self.human_se, "predictions": self.predictions,
                "separable": self.separable, "winner": self.winner}

    @classmethod
    def from_json(cls, d: dict) -> "QueryOutcome":
        return cls(proposer=d["proposer"], query=CorpusQuery(**d["query"]), human=d["human"],
                   human_se=d["human_se"], predictions=d["predictions"], separable=d["separable"],
                   winner=d["winner"])

    def summary(self) -> str:
        errs = self.errors_in_se()
        preds = ", ".join(f"{k} {v:.4f} ({errs[k]:.1f} SE off)" for k, v in self.predictions.items())
        verdict = (f"winner: {self.winner}" if self.separable
                   else f"NOT SEPARABLE (predictions differ by < {SEPARATION:g} SE)")
        return (f"proposed by {self.proposer} on {', '.join(self.query.conditions)}: "
                f"human {self.human:.4f} (SE {self.human_se:.4f}); {preds}; {verdict}")


def human_accuracy(conditions: Sequence[Condition]) -> dict[str, np.ndarray]:
    return {c.label: np.asarray(c.accuracy, dtype=float) for c in conditions}


def bootstrap_se(query: CorpusQuery, by_label: dict[str, Condition], *, n: int = N_BOOTSTRAP,
                 seed: int = 0) -> float:
    """SE of the human statistic: resample people (with replacement) within each condition."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        acc = {}
        for label in query.conditions:
            y = by_label[label].subject_correct
            rows = y[rng.integers(0, y.shape[0], size=y.shape[0])]
            acc[label] = np.nanmean(rows, axis=0)
        try:
            values.append(query.value(acc))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
    if len(values) < n // 2:
        raise ValueError("the statistic fails on most resamples of the human data")
    return float(np.std(values, ddof=1))


def evaluate_query(query: CorpusQuery, proposer: str, conditions: Sequence[Condition],
                   predicted: dict[str, dict[str, np.ndarray]], *, seed: int = 0) -> QueryOutcome:
    """Score `query` on the human data and on each theory's predicted accuracies.

    `predicted` maps theory label -> {condition label -> per-item population accuracy}.
    """
    by_label = {c.label: c for c in conditions}
    unknown = [lab for lab in query.conditions if lab not in by_label]
    if unknown:
        raise ValueError(f"unknown condition labels {unknown}")
    if any(by_label[lab].subject_correct is None for lab in query.conditions):
        raise ValueError("per-subject responses are needed for every queried condition")
    human = query.value(human_accuracy(conditions))
    se = bootstrap_se(query, by_label, seed=seed)
    preds = {theory: query.value(acc) for theory, acc in predicted.items()}
    a, b = list(preds)
    separable = abs(preds[a] - preds[b]) >= SEPARATION * se
    winner = min(preds, key=lambda t: abs(preds[t] - human)) if separable else None
    return QueryOutcome(proposer=proposer, query=query, human=human, human_se=se,
                        predictions=preds, separable=separable, winner=winner)


class DebateVerdict(Promptable):
    interpretation: str = Field(
        ..., description="What this round's outcomes show about where each theory succeeds or fails, and why.")
    target: Literal[1, 2] = Field(
        ..., description="Which theory to revise: 1 or 2 (as numbered in THEORIES).")
    verdict: Literal["new_model", "new_theory"] = Field(
        ..., description=("'new_model': keep the target theory's central claim but change its mechanism "
                          "or implementation; 'new_theory': replace it with a different account."))
    recommendation: str = Field(
        ..., description="The concrete change to make to the target, grounded in the failed queries.")


# --- prompts -------------------------------------------------------------------

QUERY_SYSTEM = """\
You are a cognitive scientist arguing for one theory of cross-situational word \
learning against another, using an existing corpus of human data.

## THE TASK AND THE DATA
On each training trial a learner hears several novel words and sees several \
novel objects, with no indication of which word goes with which; each word's \
referent is consistent across trials. At test, people choose each word's \
referent. The corpus has 47 adult conditions that vary words and objects per \
trial, word frequency, contextual diversity, temporal contiguity, filtering \
and test format. Both theories are fitted to all of it; their predictions are \
population means of per-item accuracy (people differ in ability, which the \
fit models explicitly).

## YOUR MOVE
Propose one CORPUS QUERY: a set of conditions and a statistic of their per-item \
accuracies on which YOUR theory ({advocate}) should match the human value more \
closely than the opponent ({opponent}). Good queries isolate an effect your \
theory's mechanism produces and the opponent's does not (e.g. a contrast \
between conditions that differ in one design factor, or a pattern across \
items within a condition).

The query is evaluated on the human data and on both theories' predictions. \
It counts only if the two predictions differ by at least {sep:g} standard \
errors of the human value (estimated by resampling people) - otherwise the \
data cannot decide it. The theory closer to the human value wins the query. \
Do not repeat a query already in the ledger.

## OUTPUT
{output_rules}
Return one JSON object with these fields:
{format}
"""

ARBITER_SYSTEM = """\
You are the arbiter of a debate between two theories of cross-situational \
word learning, fitted to a corpus of 47 adult conditions. Each round, each \
side proposed a corpus query - a statistic on which it expected to match \
people better - and the queries were scored against the human data.

Decide which theory to revise this round, and how: 'new_model' keeps its \
central claim and changes its mechanism or implementation; 'new_theory' \
replaces it with a different account. Base the decision on what the queries \
reveal (which effects each theory gets right or wrong, by how many standard \
errors), on the cumulative ledger, and on the cross-validated held-out NLL of \
each theory. Give one concrete, mechanistic recommendation for the target.

## OUTPUT
{output_rules}
Return one JSON object with these fields:
{format}
"""


@dataclass
class Side:
    label: str
    theory: Any          # XSLTheory
    fit: Any             # FitResult
    cv: Any              # CVResult


def _side_block(n: int, side: Side) -> str:
    params = ", ".join(f"{k}={v:.4g}" for k, v in side.fit.params.items())
    return (f"### THEORY {n}: {side.label} — held-out NLL {side.cv.heldout_nll:.1f}, "
            f"held-out wSSE {side.cv.heldout_weighted_sse:.4f}\n"
            f"Fitted parameters: {params}\n\n{side.theory.pretty_print()}")


def _condition_table(conditions: Sequence[Condition], sides: Sequence[Side],
                     predicted: dict[str, dict[str, np.ndarray]]) -> str:
    a, b = sides
    lines = [f"| label | condition | words x objects | trials | per trial | test | n_subj | human acc "
             f"| {a.label} acc | {b.label} acc | held-out NLL {a.label} minus {b.label} |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(conditions):
        wpt = np.mean([len(t) for t in c.train_words])
        opt = np.mean([len(t) for t in c.train_objects])
        test = f"{len(c.objects)}-AFC" if c.test is None else f"{len(c.test.objects[0])}-AFC"
        diff = a.cv.per_condition_nll[i] - b.cv.per_condition_nll[i]
        lines.append(
            f"| {c.label} | {c.condition} | {len(c.words)}x{len(c.objects)} | {c.n_trials} "
            f"| {wpt:.1f}x{opt:.1f} | {test} | {c.n_subj} | {np.mean(c.accuracy):.3f} "
            f"| {predicted[a.label][c.label].mean():.3f} | {predicted[b.label][c.label].mean():.3f} | {diff:+.1f} |")
    return "\n".join(lines)


def ledger_block(ledger: Sequence[QueryOutcome]) -> str:
    if not ledger:
        return "(no queries yet)"
    return "\n".join(f"- {o.summary()}" for o in ledger)


def render_query(*, advocate: Side, opponent: Side, conditions: Sequence[Condition],
                 predicted: dict[str, dict[str, np.ndarray]], ledger: Sequence[QueryOutcome],
                 failures: Sequence[str], output_rules: str) -> tuple[str, str]:
    system = QUERY_SYSTEM.format(advocate=advocate.label, opponent=opponent.label, sep=SEPARATION,
                                 output_rules=output_rules, format=CorpusQuery.instruction_format())
    parts = [
        "## THEORIES\n" + _side_block(1, advocate) + "\n\n" + _side_block(2, opponent),
        "## CONDITIONS (mean accuracy; your theory is the first prediction column)\n"
        + _condition_table(conditions, [advocate, opponent], predicted),
        "## LEDGER (all queries so far)\n" + ledger_block(ledger),
    ]
    if failures:
        parts.append("## YOUR EARLIER ATTEMPTS THIS ROUND (rejected)\n" + "\n".join(f"- {f}" for f in failures))
    return system, "\n\n".join(parts)


def render_arbitration(*, sides: Sequence[Side], conditions: Sequence[Condition],
                       predicted: dict[str, dict[str, np.ndarray]], round_outcomes: Sequence[QueryOutcome],
                       ledger: Sequence[QueryOutcome], output_rules: str) -> tuple[str, str]:
    system = ARBITER_SYSTEM.format(output_rules=output_rules, format=DebateVerdict.instruction_format())
    parts = [
        "## THEORIES\n" + "\n\n".join(_side_block(i + 1, s) for i, s in enumerate(sides)),
        "## THIS ROUND'S QUERIES\n" + "\n".join(f"- {o.summary()}\n  claim: {o.query.claim}"
                                                for o in round_outcomes),
        "## EARLIER ROUNDS\n" + ledger_block([o for o in ledger if all(o is not r for r in round_outcomes)]),
        "## CONDITIONS\n" + _condition_table(conditions, list(sides), predicted),
    ]
    return system, "\n\n".join(parts)
