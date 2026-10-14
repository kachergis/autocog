"""NumPy XSL ports vs closed-form values and vs R (XSLmodels @ the pinned commit).

The R fixtures come from `scripts/xsl/export_corpus.R` (reference_predictions.json
and the v05 incumbent fits in incumbents.json), so these tests need no R.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.xsl import models, scoring
from src.xsl.corpus import DATA_DIR, Condition, MAFCTest, load_corpus

# Ports must reproduce R to well inside the 1e-8 the proposal asks for;
# observed worst case is ~2e-13.
ATOL = 1e-10


@pytest.fixture(scope="module")
def corpus() -> list[Condition]:
    return load_corpus()


@pytest.fixture(scope="module")
def reference() -> list[dict]:
    return json.loads((DATA_DIR / "reference_predictions.json").read_text())


@pytest.fixture(scope="module")
def incumbents() -> dict:
    return json.loads((DATA_DIR / "incumbents.json").read_text())


# --- closed-form checks ------------------------------------------------------

def test_entropy_of_uniform_is_log2_n() -> None:
    assert models.shannon_entropy(np.full(8, 0.3)) == pytest.approx(3.0)
    assert models.shannon_entropy(np.array([0.0, 2.0, 0.0])) == 0.0


def test_update_known_seeds_trial_and_familiar_pairs() -> None:
    m = np.zeros((3, 3))
    m[0, 0] = 1.0  # word 0 / object 0 are familiar
    models.update_known(m, np.array([1]), np.array([1]))
    expected = np.array([
        [1.0, 0.01, 0.0],   # familiar word 0 x trial object 1
        [0.01, 0.01, 0.0],  # trial pair, then trial word 1 x familiar object 0
        [0.0, 0.0, 0.0],
    ])
    np.testing.assert_array_equal(m, expected)


def test_baseline_counts_cooccurrences() -> None:
    c = Condition(label="t", condition="2x2", train_words=((1, 2), (1, 3)),
                  train_objects=((1, 2), (1, 3)))
    perf, m = scoring.run(models.baseline, {}, c)
    np.testing.assert_array_equal(m, [[2, 1, 1], [1, 1, 0], [1, 0, 1]])
    np.testing.assert_allclose(perf, [0.5, 0.5, 0.5])


def test_uncfam_single_trial_splits_X_evenly() -> None:
    # one 2x2 trial from scratch: every pair is seeded at .01 with identical
    # entropy, so X is split equally over the four pairs
    c = Condition(label="t", condition="2x2", train_words=((1, 2),), train_objects=((1, 2),))
    _, m = scoring.run(models.uncfam, {"X": 1.0, "B": 3.0, "C": 0.5}, c)
    np.testing.assert_allclose(m, np.full((2, 2), 0.01 * 0.5 + 0.25))


def test_get_perf_luce_and_chance() -> None:
    m = np.array([[3.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(scoring.get_perf(m), [0.75, 1 / 3])
    np.testing.assert_allclose(scoring.get_perf(m, d=2), [0.9, 1 / 3])
    with pytest.raises(ValueError):
        scoring.get_perf(np.ones((3, 2)))


def test_mafc_test_matches_labels_and_explicit_targets() -> None:
    m = np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 2.0]])
    words, objects = (10, 20), (10, 20, 30)
    test = MAFCTest(words=(10, 20, 99), objects=((10, 20), (20, 30), (10, 30)))
    np.testing.assert_allclose(scoring.mafc_test(m, words, objects, test), [0.75, 0.5, math.nan])
    explicit = MAFCTest(words=(20,), objects=((20, 30),), correct=(30,))
    np.testing.assert_allclose(scoring.mafc_test(m, words, objects, explicit), [0.5])
    novel = MAFCTest(words=(99,), objects=((10, 20, 30),), correct=(10,))
    np.testing.assert_allclose(scoring.mafc_test(m, words, objects, novel), [1 / 3])


# --- parity with R -------------------------------------------------------------

def test_reference_runs_cover_ported_models() -> None:
    reference = json.loads((DATA_DIR / "reference_predictions.json").read_text())
    assert {r["constructor"] for r in reference} == set(models.LEARNERS)


@pytest.mark.parametrize("run_idx", range(4))
def test_port_matches_r_on_every_condition(corpus, reference, run_idx) -> None:
    run = reference[run_idx]
    learner = models.LEARNERS[run["constructor"]]
    assert run["fixed"].get("variant", "entropy") == "entropy"
    for cond, r in zip(corpus, run["corpus"], strict=True):
        perf, m = scoring.run(learner, run["params"], cond)
        assert [str(w) for w in cond.words] == r["matrix_rows"]
        assert [str(o) for o in cond.objects] == r["matrix_cols"]
        np.testing.assert_allclose(m, np.array(r["matrix"], dtype=float), rtol=ATOL, atol=ATOL,
                                   err_msg=f"matrix, condition {cond.label}")
        np.testing.assert_allclose(perf, np.array(r["perf"], dtype=float), rtol=0, atol=ATOL,
                                   err_msg=f"perf, condition {cond.label}")


@pytest.mark.parametrize("name,learner", [("kachergis", models.uncfam), ("baseline", models.baseline)])
def test_port_reproduces_v05_group_fit(corpus, incumbents, name, learner) -> None:
    inc = incumbents[name]
    assert scoring.weighted_sse(learner, inc["params"], corpus) == pytest.approx(inc["sse_v05"], abs=1e-12)
    for cond, v05 in zip(corpus, inc["v05_perf"], strict=True):
        perf, _ = scoring.run(learner, inc["params"], cond)
        np.testing.assert_allclose(perf, np.array(v05, dtype=float), rtol=0, atol=ATOL,
                                   err_msg=f"condition {cond.label}")


def test_uncfam_gamma_one_is_uncfam(corpus) -> None:
    params = {"X": 0.2, "B": 4.0, "C": 0.95}
    for cond in corpus[:5]:
        _, m1 = scoring.run(models.uncfam, params, cond)
        _, m2 = scoring.run(models.uncfam, {**params, "gamma": 1.0}, cond)
        np.testing.assert_array_equal(m1, m2)
