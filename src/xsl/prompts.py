"""Prompts for the XSL theory generator and its critic (mode B: full corpus).

Rendering only: no LLM calls, no fitting. Everything numeric shown to the LLM
comes from `src.xsl.fitting` results passed in by the generator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import Field

from src.promptable import Promptable
from src.xsl.corpus import Condition
from src.xsl.fitting import CVResult, FitResult
from src.xsl.theory import XSLTheory


class CritiqueVerdict(Promptable):
    interpretation: str = Field(
        ...,
        description="Where the candidate fits the human data well or badly, and which mechanism is responsible.",
    )
    verdict: Literal["regenerate", "continue"] = Field(
        ...,
        description="'continue' to accept the candidate as this loop's answer, or 'regenerate' to ask for a revision.",
    )
    rationale: str = Field(
        ...,
        description="If 'regenerate': the concrete change to the mechanism or parameters to try next. If 'continue': why it is good enough.",
    )


@dataclass
class Entry:
    """One theory on the leaderboard: its full-corpus fit and its cross-validation."""

    label: str
    theory: XSLTheory
    fit: FitResult
    cv: CVResult | None = None

    @property
    def score(self) -> float:
        """Held-out NLL (lower is better); entries not yet cross-validated sort last."""
        return self.cv.heldout_nll if self.cv is not None else math.inf


@dataclass
class Iteration:
    """One critique iteration inside a generator loop."""

    candidate: XSLTheory | None
    fit: FitResult | None
    accepted: bool
    critique: CritiqueVerdict | None = None
    error: str | None = None
    cv: CVResult | None = None

    @property
    def bic(self) -> float:
        return self.fit.bic if self.fit is not None else math.inf

    @property
    def score(self) -> float:
        """Held-out NLL: what the accept gate compares."""
        return self.cv.heldout_nll if self.cv is not None else math.inf


DOMAIN = """\
## THE TASK: CROSS-SITUATIONAL WORD LEARNING
On each training trial a learner hears several novel words and sees several \
novel objects, with no indication of which word goes with which object. Each \
word has a consistent referent across trials, so its meaning can only be \
learned by tracking word-object co-occurrence across situations. After \
training, learning is tested per word: the learner hears a word and chooses \
its referent, either among ALL objects (n-AFC) or among a few candidates \
(m-AFC). The corpus holds 47 experimental conditions from published adult \
studies, which vary the number of words and objects per trial, word \
frequency, contextual diversity, temporal contiguity, filtering, and test \
format.

## WHAT A THEORY MUST PROVIDE
A prose claim about the learning mechanism, plus one Python function:

```python
def learn(params, trials, n_words, n_objects):
    # params: dict of parameter values (floats), keyed by the names you declare
    # trials: list of (words, objects) pairs, in training order; each is a 1-D
    #         numpy int array of 0-based indices (words in 0..n_words-1,
    #         objects in 0..n_objects-1)
    # returns: numpy array, shape (n_words, n_objects), of non-negative
    #          word-object association strengths after training
```

Rules for `learn`:
  * `np` (numpy) is the only module available; do not import anything.
  * It must be deterministic (no randomness) and return finite, non-negative values.
  * It must be fast: one pass over all 47 conditions must take well under 2 seconds,
    because the fitter calls it thousands of times.
  * The correct referent of word i is NOT given to `learn`; it only sees co-occurrences.

Test choices are made by the Luce rule from your matrix: P(choose object j for \
word i) = m[i, j] / sum_k m[i, k] over the candidate objects. If you declare a \
parameter named `ch_dec`, the rule becomes m[i, j]**ch_dec / sum_k m[i, k]**ch_dec. \
Do not implement the choice rule yourself.

## PARAMETERS
`parameters` maps each name to a string interval "[lo, hi]" (fitted to the \
data) or a bare number (held fixed). Add a parameter only if it captures \
something that generalises across conditions (see scoring below). Every \
theory automatically gets one extra fitted parameter, `lapse`: the rate of \
pure guessing (uniform over the choices) at test. Do not declare `lapse` \
yourself, and do not model random guessing inside `learn`.

## HOW THEORIES ARE SCORED
Each theory is FITTED to all 47 conditions at once (one parameter set for \
everyone) by differential evolution, minimising the negative log-likelihood \
(NLL) of the human test responses: {objective_text}

The SCORE is the held-out NLL from 5-fold cross-validation over conditions: \
the theory is fitted to four fifths of the conditions and must predict the \
human choices in the remaining fifth, for each fifth in turn (always the same \
split), and the five held-out NLLs are summed. Lower is better. Extra \
parameters that only fit the training conditions do not help here and \
usually hurt. Also shown, for reference only: BIC = 2*NLL + k*ln(N) of the \
full-corpus fit (k free parameters incl. `lapse`, N = {n_choices} choices) and \
the n_subj-weighted SSE on per-word accuracy (the xsl-bakeoff objective).
"""

GENERATOR_SYSTEM = """\
You are a cognitive scientist and expert Python programmer working on theories \
of cross-situational word learning. Your job is to propose a NEW theory, with \
an executable learner, that predicts the human data better (lower held-out \
NLL) than every theory on the leaderboard.

{domain}
## ACCEPT GATE
You work in a loop. Each proposal is fitted and cross-validated; it is \
ACCEPTED only if its held-out NLL is strictly lower than that of the best \
accepted candidate so far in this loop. `## CURRENT BEST CANDIDATE` always shows the best accepted candidate, \
never a rejected one, so build on it. A critic comments on each candidate; \
the `## LOOP HISTORY` block shows which of its suggestions helped (ACCEPTED) \
and which did not (REJECTED).

Leaderboard theories are for inspiration only: do not resubmit one of them \
or a cosmetic variant. A new mechanism that fits better is the goal.

## OUTPUT
{output_rules}
Return one JSON object with these fields:
{format}
"""

CRITIC_SYSTEM = """\
You are a critical reviewer of cognitive models of cross-situational word \
learning. You are shown a candidate theory, its fit to the human corpus, and \
the leaderboard. Diagnose where and why it misfits, and decide whether the \
loop should stop with this candidate ('continue') or revise it ('regenerate'). \
Say 'continue' only if the candidate was ACCEPTED and you see no concrete, \
promising revision; otherwise give one specific mechanistic change to try.

{domain}
## OUTPUT
{output_rules}
Return one JSON object with these fields:
{format}
"""

# Sonnet 5.5 sometimes opened its reply with invented tool calls
# (`<invoke name="bash">...`); once that ran on to the 128k-token output cap.
OUTPUT_RULES = (
    "You have no tools and cannot run code: do not write tool calls, XML tags or "
    "`<invoke>` blocks. Reply with the JSON object only — no prose before or after it."
)


OBJECTIVE_TEXT = {
    "binomial": (
        "for every test item, how many people chose the correct referent out of "
        "how many were tested (binomial). The score is about item-level ACCURACY: "
        "which wrong object people chose does not count."
    ),
    "multinomial": (
        "multinomial over objects for n-AFC tests (so WHICH wrong object people "
        "choose matters, not just accuracy), binomial correct/incorrect for m-AFC tests."
    ),
}


def domain_text(n_choices: int, objective: str) -> str:
    return DOMAIN.format(n_choices=n_choices, objective_text=OBJECTIVE_TEXT[objective])


def _design_row(c: Condition) -> str:
    wpt = np.mean([len(t) for t in c.train_words])
    opt = np.mean([len(t) for t in c.train_objects])
    freq = np.bincount(np.concatenate([w for w, _ in c.positional_trials])).mean()
    test = "n-AFC" if c.test is None else f"{len(c.test.objects[0])}-AFC"
    return (
        f"| {c.label} | {c.condition} | {len(c.words)}x{len(c.objects)} | {c.n_trials} "
        f"| {wpt:.1f}x{opt:.1f} | {freq:.1f} | {test} | {c.n_subj} | {np.mean(c.accuracy):.3f} |"
    )


def corpus_table(conditions: Sequence[Condition]) -> str:
    header = (
        "| label | condition | words x objects | trials | words x objects per trial "
        "| mean word freq | test | n_subj | human acc |\n|---|---|---|---|---|---|---|---|---|"
    )
    return header + "\n" + "\n".join(_design_row(c) for c in conditions)


def _params(fit: FitResult) -> str:
    vals = ", ".join(f"{k}={v:.4g}" for k, v in fit.params.items()) or "(none)"
    return vals + (f" — at bound: {', '.join(fit.at_bound)}" if fit.at_bound else "")


def _held_out(cv: CVResult | None) -> str:
    if cv is None:
        return "n/a | n/a"
    return f"{cv.heldout_nll:.1f} | {cv.heldout_weighted_sse:.4f}"


def leaderboard_table(entries: Sequence[Entry]) -> str:
    rows = sorted(entries, key=lambda e: e.score)
    lines = ["| rank | label | name | k | held-out NLL (score) | held-out wSSE | BIC (full fit) "
             "| fitted parameters (full fit) |",
             "|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(rows, 1):
        lines.append(
            f"| {i} | {e.label} | {e.theory.name} | {e.fit.n_free} | {_held_out(e.cv)} "
            f"| {e.fit.bic:.1f} | {_params(e.fit)} |"
        )
    return "\n".join(lines)


def leaderboard_details(entries: Sequence[Entry]) -> str:
    return "\n\n".join(
        f"### {e.label}: {e.theory.name}\n{e.theory.description}" for e in sorted(entries, key=lambda e: e.score)
    )


def _per_condition_nll(fit: FitResult, cv: CVResult | None) -> list[float]:
    return cv.per_condition_nll if cv is not None else fit.evaluation.per_condition_nll


def fit_diagnostics(fit: FitResult, cv: CVResult | None, conditions: Sequence[Condition],
                    reference: Entry | None) -> str:
    """Per-condition fit, worst first: held-out NLL when cross-validated (else
    full-fit NLL), ranked by its excess over the reference theory if given,
    otherwise by NLL per human choice (raw NLL grows with condition size)."""
    ev = fit.evaluation
    nll = _per_condition_nll(fit, cv)
    ref = _per_condition_nll(reference.fit, reference.cv) if reference is not None else None
    rows = []
    for i, c in enumerate(conditions):
        key = nll[i] - ref[i] if ref else nll[i] / c.response_matrix.counts.sum()
        rows.append((key, c, ev.per_condition_perf[i], nll[i]))
    rows.sort(key=lambda r: -r[0])
    kind = "held-out NLL" if cv is not None else "NLL"
    last = f"{kind} minus {reference.label}'s" if reference is not None else f"{kind} per choice"
    lines = [f"| label | condition | human acc | model acc (full fit) | {kind} | {last} |",
             "|---|---|---|---|---|---|"]
    for key, c, perf, nll in rows:
        shown = f"{key:+.1f}" if reference is not None else f"{key:.3f}"
        lines.append(
            f"| {c.label} | {c.condition} | {np.mean(c.accuracy):.3f} | {np.nanmean(perf):.3f} "
            f"| {nll:.1f} | {shown} |"
        )
    return "\n".join(lines)


def history_block(history: Sequence[Iteration]) -> str:
    if not history:
        return "(first iteration)"
    out = []
    for i, it in enumerate(history):
        if it.error:
            out.append(f"- iter {i}: FAILED — {it.error}")
            continue
        tag = "ACCEPTED" if it.accepted else "REJECTED"
        line = f"- iter {i}: {it.candidate.name} — held-out NLL {it.score:.1f} (BIC {it.bic:.1f}) — {tag}"
        if it.critique is not None:
            line += f"\n  critic: {it.critique.rationale}"
        out.append(line)
    return "\n".join(out)


def render_generation(
    *,
    conditions: Sequence[Condition],
    leaderboard: Sequence[Entry],
    guide: str,
    best: Iteration | None,
    history: Sequence[Iteration],
    n_choices: int,
    objective: str = "binomial",
) -> tuple[str, str]:
    system = GENERATOR_SYSTEM.format(
        domain=domain_text(n_choices, objective), format=XSLTheory.instruction_format(),
        output_rules=OUTPUT_RULES,
    )
    reference = min(leaderboard, key=lambda e: e.score) if leaderboard else None
    parts = [
        f"## GOAL\n{guide}",
        f"## CORPUS (47 conditions)\n{corpus_table(conditions)}",
        f"## LEADERBOARD (ranked by held-out NLL; lower is better)\n{leaderboard_table(leaderboard)}",
        f"## LEADERBOARD THEORIES\n{leaderboard_details(leaderboard)}",
    ]
    if reference is not None:
        parts.append(
            f"## BEST LEADERBOARD THEORY: {reference.label}\n{reference.theory.pretty_print()}\n"
            f"Per-condition fit, worst first:\n{fit_diagnostics(reference.fit, reference.cv, conditions, None)}"
        )
    if best is not None and best.fit is not None:
        parts.append(
            f"## CURRENT BEST CANDIDATE (this loop) — held-out NLL {best.score:.1f}, "
            f"BIC {best.fit.bic:.1f}, params {_params(best.fit)}\n"
            f"{best.candidate.pretty_print()}\n"
            f"Per-condition fit, worst first:\n{fit_diagnostics(best.fit, best.cv, conditions, reference)}"
        )
    parts.append(f"## LOOP HISTORY\n{history_block(history)}")
    return system, "\n\n".join(parts)


def render_critique(
    *,
    conditions: Sequence[Condition],
    leaderboard: Sequence[Entry],
    candidate: Iteration,
    history: Sequence[Iteration],
    n_choices: int,
    objective: str = "binomial",
) -> tuple[str, str]:
    system = CRITIC_SYSTEM.format(
        domain=domain_text(n_choices, objective), format=CritiqueVerdict.instruction_format(),
        output_rules=OUTPUT_RULES,
    )
    reference = min(leaderboard, key=lambda e: e.score) if leaderboard else None
    tag = "ACCEPTED" if candidate.accepted else "REJECTED (not better than this loop's best)"
    parts = [
        f"## CANDIDATE — held-out NLL {candidate.score:.1f}, BIC {candidate.fit.bic:.1f}, "
        f"wSSE (full fit) {candidate.fit.evaluation.weighted_sse:.4f} — {tag}\n"
        f"Fitted parameters: {_params(candidate.fit)}\n\n{candidate.candidate.pretty_print()}",
        f"## PER-CONDITION FIT, WORST FIRST\n{fit_diagnostics(candidate.fit, candidate.cv, conditions, reference)}",
        f"## CORPUS\n{corpus_table(conditions)}",
        f"## LEADERBOARD\n{leaderboard_table(leaderboard)}",
        f"## LOOP HISTORY\n{history_block(history)}",
    ]
    return system, "\n\n".join(parts)
