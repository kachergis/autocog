"""NumPy reference ports of XSLmodels learners.

Each learner maps a training sequence to a word x object association matrix:

    learner(params, trials, n_words, n_objects) -> np.ndarray  # (n_words, n_objects)

`trials` is `Condition.positional_trials`: per trial, the 0-based word rows and
object columns shown. Scoring the matrix against a test is `src.xsl.scoring`'s
job, exactly as XSLmodels splits `*_model()` from `get_perf()` / `mafc_test()`.

These ports are checked against R output from the pinned XSLmodels build
(`tests/test_xsl_parity.py`), so they double as the executable spec that
LLM-written learners are compared to. Source of each: XSLmodels @ the commit in
`data/xsl/manifest.json`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

Trials = Sequence[tuple[np.ndarray, np.ndarray]]

# XSLmodels' `update_known` start value for newly encountered pairs.
START_VAL = 0.01


def shannon_entropy(p: np.ndarray) -> float:
    """Entropy (bits) of `p` after normalising by its total; zeros are skipped.

    XSLmodels' `shannon_entropy` returns NA for a negative or all-zero vector;
    neither can reach the uncfam update (rows/columns on the trial were just
    filled by `update_known`), so this raises instead of propagating NaN.
    """
    s = p.sum()
    if p.min() < 0 or s <= 0:
        raise ValueError("entropy undefined for negative or all-zero associations")
    q = p[p > 0] / s
    return float(-(q * np.log2(q)).sum())


def update_known(m: np.ndarray, words: np.ndarray, objects: np.ndarray,
                 start_val: float = START_VAL) -> np.ndarray:
    """XSLmodels `update_known`, in place: seed never-seen pairs at `start_val`.

    Every (word, object) pair on the trial that is still 0 gets `start_val`;
    then each trial word gets `start_val` with every *familiar* object (one
    with any association so far) it has no association with, and each trial
    object likewise with every familiar word. Familiarity is computed once,
    after the trial's own pairs are seeded — as in R.
    """
    block = m[np.ix_(words, objects)]
    block[block == 0] = start_val
    m[np.ix_(words, objects)] = block
    fam_objects = m.sum(axis=0) > 0
    fam_words = m.sum(axis=1) > 0
    for w in words:
        m[w, fam_objects & (m[w, :] == 0)] = start_val
    for o in objects:
        m[fam_words & (m[:, o] == 0), o] = start_val
    return m


def baseline(params: dict[str, Any], trials: Trials, n_words: int, n_objects: int) -> np.ndarray:
    """Co-occurrence counting: +1 to every word-object pair on each trial."""
    m = np.zeros((n_words, n_objects))
    for words, objects in trials:
        m[np.ix_(words, objects)] += 1
    return m


def uncfam(params: dict[str, Any], trials: Trials, n_words: int, n_objects: int) -> np.ndarray:
    """Kachergis, Yu & Shiffrin (2012) uncertainty- and familiarity-biased learner.

    Entropy variant (bakeoff name `kachergis`). On each trial a fixed amount
    of associative weight `X` is split over the trial's word-object pairs in
    proportion to

        assoc(w, o) ** gamma * exp(B * H(w)) * exp(B * H(o))

    — familiarity (the pair's current strength) times uncertainty (the
    entropy of the word's row and the object's column) — after all
    associations decay by `C`. `gamma` defaults to 1, which is plain uncfam;
    a free `gamma` is XSLmodels' `uncfam_gamma`.
    """
    X, B, C = params["X"], params["B"], params["C"]
    gamma = params.get("gamma", 1.0)
    m = np.zeros((n_words, n_objects))
    for words, objects in trials:
        update_known(m, words, objects)
        assocs = m[np.ix_(words, objects)]
        h_w = np.array([shannon_entropy(m[w, :]) for w in words])
        h_o = np.array([shannon_entropy(m[:, o]) for o in objects])
        familiarity = assocs if gamma == 1.0 else assocs ** gamma
        terms = familiarity * np.outer(np.exp(B * h_w), np.exp(B * h_o))
        m *= C
        m[np.ix_(words, objects)] += X * terms / terms.sum()
    return m


LEARNERS = {
    "baseline": baseline,
    "uncfam": uncfam,
}
