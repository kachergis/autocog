"""Turn an association matrix into predicted test accuracy, and score it.

Ports of XSLmodels' `get_perf` (n-AFC over all objects, diagonal convention)
and `mafc_test` / `predict_referent` (explicit m-AFC test, matched by label),
plus the run-and-score wrapper that `xsl_run` provides in R.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from src.xsl.corpus import Condition, MAFCTest

Learner = Callable[..., np.ndarray]


def get_perf(m: np.ndarray, d: float | None = None) -> np.ndarray:
    """Luce choice over all objects: P(word i -> object i) = m[i, i] / sum_j m[i, j].

    `d` is XSLmodels' optional choice exponent (`ch_dec`). A word with no
    associations at all is at chance (1 / n_objects). Only defined for
    n_words <= n_objects: R's `diag(m) / rowSums(m)` silently recycles when
    there are more words than objects, so any such design must use an
    explicit `MAFCTest`.
    """
    n_words, n_objects = m.shape
    if n_words > n_objects:
        raise ValueError(
            f"diagonal scoring needs n_words <= n_objects (got {n_words} x {n_objects}); "
            "give the condition an explicit MAFCTest"
        )
    m = np.maximum(m, 0.0) ** (1.0 if d is None else d)
    total = m.sum(axis=1)
    diag = m[np.arange(n_words), np.arange(n_words)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, diag / total, 1.0 / n_objects)


def predict_referent(m: np.ndarray, words: Sequence[Any], objects: Sequence[Any],
                     word: Any, candidates: Sequence[Any]) -> np.ndarray:
    """P(choose each candidate | probe word): the word's row over the candidates.

    Labels are matched against the matrix's word / object labels; a candidate
    the learner never saw contributes 0, and a novel word (or a row with no
    mass on any candidate) is at chance over the candidates.
    """
    k = len(candidates)
    uniform = np.full(k, 1.0 / k)
    try:
        row_idx = list(words).index(word)
    except ValueError:
        return uniform
    obj_pos = {o: j for j, o in enumerate(objects)}
    M = np.maximum(m, 0.0)
    row = np.array([M[row_idx, obj_pos[c]] if c in obj_pos else 0.0 for c in candidates])
    s = row.sum()
    return row / s if s > 0 else uniform


def mafc_test(m: np.ndarray, words: Sequence[Any], objects: Sequence[Any], test: MAFCTest) -> np.ndarray:
    """P(correct) per test trial; the target is `test.correct[i]` or the probe word's label."""
    out = np.empty(len(test))
    for i, (word, candidates) in enumerate(zip(test.words, test.objects)):
        probs = predict_referent(m, words, objects, word, candidates)
        target = word if test.correct is None else test.correct[i]
        out[i] = probs[list(candidates).index(target)] if target in candidates else np.nan
    return out


def predict_accuracy(m: np.ndarray, condition: Condition, d: float | None = None) -> np.ndarray:
    """Predicted per-item accuracy for `condition` from its final association matrix."""
    if condition.test is not None:
        return mafc_test(m, condition.words, condition.objects, condition.test)
    return get_perf(m, d=d)


def run(learner: Learner, params: dict[str, Any], condition: Condition) -> tuple[np.ndarray, np.ndarray]:
    """Train `learner` on `condition` and score it: returns (per-item accuracy, matrix)."""
    m = learner(params, condition.positional_trials, len(condition.words), len(condition.objects))
    return predict_accuracy(m, condition, d=params.get("ch_dec")), m


def sse(perf: np.ndarray, condition: Condition) -> float:
    if condition.accuracy is None:
        raise ValueError(f"{condition.label} has no human accuracy to score against")
    return float(((perf - condition.accuracy) ** 2).sum())


def weighted_sse(learner: Learner, params: dict[str, Any], conditions: Sequence[Condition]) -> float:
    """The xsl-bakeoff group objective: per-condition SSE, averaged with weights n_subj."""
    terms = np.array([sse(run(learner, params, c)[0], c) for c in conditions])
    weights = np.array([c.n_subj for c in conditions], dtype=float)
    return float((terms * weights).sum() / weights.sum())
