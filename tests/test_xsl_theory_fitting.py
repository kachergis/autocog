"""XSLTheory parsing/validation and the group-likelihood fitter."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import binom, multinomial

from src.xsl import fitting, models, scoring
from src.xsl.corpus import Condition, MAFCTest, ResponseMatrix, load_corpus
from src.xsl.theory import XSLTheory, parse_parameters

UNIFORM = "def learn(params, trials, n_words, n_objects):\n    return np.ones((n_words, n_objects))\n"


@pytest.fixture(scope="module")
def corpus() -> list[Condition]:
    return load_corpus()


def _toy(counts: np.ndarray, test: MAFCTest | None = None) -> Condition:
    """Two words, two objects, one 2x2 trial; response counts supplied."""
    return Condition(
        label="toy", condition="2x2", train_words=((1, 2),), train_objects=((1, 2),),
        test=test, accuracy=np.zeros(counts.shape[0]), n_subj=int(counts.sum(axis=1).max()),
        response_matrix=ResponseMatrix(items=("1", "2"), objects=("1", "2"), counts=counts),
    )


# --- parameters and validation ------------------------------------------------

def test_parse_parameters() -> None:
    free, fixed = parse_parameters({"a": "[0, 1]", "b": [2, 5], "c": "3.5", "d": 1})
    assert free == {"a": (0.0, 1.0), "b": (2.0, 5.0)}
    assert fixed == {"c": 3.5, "d": 1.0}
    for bad in ({"a": "[1, 0]"}, {"a": "{1, 2}"}, {"a": "[0, inf]"}):
        with pytest.raises(ValueError):
            parse_parameters(bad)


def test_seed_yamls_match_reference_ports(corpus) -> None:
    params = {"X": 0.3, "B": 3.0, "C": 0.95}
    for name, port in [("kachergis", models.uncfam), ("baseline", models.baseline)]:
        theory = XSLTheory.from_yaml(f"theories/xsl/{name}.yaml")
        for c in corpus:
            # learner() shuffles indices, which can change float summation order
            np.testing.assert_allclose(scoring.run(theory.learner(), params, c)[1],
                                       scoring.run(port, params, c)[1], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("source,match", [
    ("def learn(params, trials, n_words, n_objects):\n    return np.ones((n_words, 1))\n", "shape"),
    ("def learn(params, trials, n_words, n_objects):\n    return -np.ones((n_words, n_objects))\n", "negative"),
    ("def learn(params, trials, n_words, n_objects):\n    return np.random.rand(n_words, n_objects)\n", "deterministic"),
])
def test_check_rejects_bad_learners(corpus, source, match) -> None:
    with pytest.raises(ValueError, match=match):
        XSLTheory(theory="t", learn=source).check(corpus[:3])


def test_check_rejects_slow_learners(corpus) -> None:
    with pytest.raises(ValueError, match="fast enough"):
        XSLTheory.from_yaml("theories/xsl/kachergis.yaml").check(corpus, max_seconds=1e-6)


def test_index_identity_cannot_be_exploited(corpus) -> None:
    # "word i goes with object i" is true of the corpus's sorted labels; a
    # learner that bets on it must not beat chance once indices are shuffled
    cheat = XSLTheory(theory="t", learn=(
        "def learn(params, trials, n_words, n_objects):\n"
        "    return np.eye(n_words, n_objects) + 1e-3\n"))
    c = next(c for c in corpus if c.test is None)
    perf, _ = scoring.run(cheat.learner(), {}, c)
    unshuffled, _ = scoring.run(cheat._learn_impl, {}, c)
    assert unshuffled.mean() > 0.9
    assert perf.mean() < 0.2


def test_learn_must_be_defined() -> None:
    with pytest.raises(ValueError):
        XSLTheory(theory="t", learn="def other():\n    pass\n")


def test_yaml_round_trip(tmp_path) -> None:
    theory = XSLTheory.from_yaml("theories/xsl/kachergis.yaml")
    theory.to_yaml(tmp_path / "t.yaml")
    again = XSLTheory.from_yaml(tmp_path / "t.yaml")
    assert again.model_dump() == theory.model_dump()


# --- likelihood: closed form -----------------------------------------------------

def test_nafc_nll_is_multinomial_with_lapse() -> None:
    counts = np.array([[7.0, 3.0], [2.0, 8.0]])
    c = _toy(counts)
    m = np.array([[3.0, 1.0], [1.0, 1.0]])
    lapse = 0.2
    p = (1 - lapse) * (m / m.sum(axis=1, keepdims=True)) + lapse / 2
    expected = -sum(multinomial.logpmf(counts[i], n=counts[i].sum(), p=p[i]) for i in range(2))
    assert fitting.condition_nll(m, c, lapse=lapse, objective="multinomial") == pytest.approx(expected, rel=1e-12)


def test_nafc_nll_is_binomial_on_accuracy_by_default() -> None:
    # word i's referent is object i: only the diagonal counts, out of each row's total
    counts = np.array([[7.0, 3.0], [2.0, 8.0]])
    m = np.array([[3.0, 1.0], [1.0, 1.0]])
    lapse = 0.2
    p = (1 - lapse) * np.array([0.75, 0.5]) + lapse / 2
    expected = -(binom.logpmf(7, 10, p[0]) + binom.logpmf(8, 10, p[1]))
    assert fitting.condition_nll(m, _toy(counts), lapse=lapse) == pytest.approx(expected, rel=1e-12)
    # moving errors between wrong objects changes the multinomial NLL, not the binomial one
    wider = _toy(np.array([[7.0, 3.0, 0.0], [2.0, 8.0, 0.0]]))
    wider = Condition(label="w", condition="", train_words=((1, 2),), train_objects=((1, 2, 3),),
                      accuracy=np.zeros(2), n_subj=10, response_matrix=replace(
                          wider.response_matrix, objects=("1", "2", "3")))
    moved = replace(wider, response_matrix=replace(wider.response_matrix,
                                                   counts=np.array([[7.0, 0.0, 3.0], [0.0, 8.0, 2.0]])))
    m3 = np.array([[3.0, 1.0, 0.5], [1.0, 1.0, 0.5]])
    assert fitting.condition_nll(m3, wider) == pytest.approx(fitting.condition_nll(m3, moved), rel=1e-12)
    assert fitting.condition_nll(m3, wider, objective="multinomial") != pytest.approx(
        fitting.condition_nll(m3, moved, objective="multinomial"), rel=1e-6)


def test_unknown_objective_rejected() -> None:
    with pytest.raises(ValueError):
        fitting.FitBudget(objective="sse")


def test_mafc_nll_is_binomial_on_target_count() -> None:
    # extra count on a non-candidate (as in corpus conditions 211/214) only
    # changes n, never the target's probability
    counts = np.array([[6.0, 4.0], [5.0, 5.0]])
    test = MAFCTest(words=(1, 2), objects=((1, 2), (1, 2)), correct=(1, 2))
    c = _toy(counts, test)
    m = np.array([[3.0, 1.0], [1.0, 1.0]])
    lapse = 0.2
    p1 = (1 - lapse) * 0.75 + lapse / 2
    p2 = (1 - lapse) * 0.5 + lapse / 2
    expected = -(binom.logpmf(6, 10, p1) + binom.logpmf(5, 10, p2))
    assert fitting.condition_nll(m, c, lapse=lapse) == pytest.approx(expected, rel=1e-12)


def test_single_candidate_trial_is_certain() -> None:
    # one candidate: p = 1, so all-correct costs nothing and any error is impossible
    test = MAFCTest(words=(1, 2), objects=((1,), (2,)), correct=(1, 2))
    m = np.array([[3.0, 1.0], [1.0, 1.0]])
    assert fitting.condition_nll(m, _toy(np.array([[10.0, 0.0], [0.0, 10.0]]), test)) == pytest.approx(0.0)
    assert fitting.condition_nll(m, _toy(np.array([[9.0, 1.0], [0.0, 10.0]]), test)) == math.inf


def test_zero_parameter_cv_equals_full_evaluation(corpus) -> None:
    conds = corpus[:6]
    cv = fitting.cross_validate(models.baseline, {}, conds, k=3, fixed={"lapse": 0.3}, fit_lapse=False)
    ev = fitting.evaluate(models.baseline, {"lapse": 0.3}, conds)
    assert cv.heldout_nll == pytest.approx(ev.nll, rel=1e-12)
    assert cv.heldout_weighted_sse == pytest.approx(ev.weighted_sse, rel=1e-12)


def test_bic_and_aic(corpus) -> None:
    r = fitting.fit(models.baseline, {}, corpus[:4], fit_lapse=False)
    assert r.n_free == 0 and r.params == {}
    assert r.bic == pytest.approx(2 * r.nll)
    assert r.n_choices == fitting.total_choices(corpus[:4])


def test_lapse_is_fitted_and_counted(corpus) -> None:
    r = fitting.fit(models.baseline, {}, corpus[:4], budget=fitting.FitBudget(popsize=6, maxiter=20))
    assert r.n_free == 1 and set(r.params) == {"lapse"}
    lo, hi = fitting.LAPSE_BOUNDS
    assert lo <= r.params["lapse"] <= hi
    assert r.bic == pytest.approx(2 * r.nll + math.log(r.n_choices))
    # fitting the guessing rate can only help over the floor value
    floor = fitting.evaluate(models.baseline, {"lapse": lo}, corpus[:4]).nll
    assert r.nll < floor


def test_reported_accuracy_includes_guessing(corpus) -> None:
    c = corpus[0]  # n-AFC over all objects
    raw, _ = scoring.run(models.baseline, {}, c)
    ev = fitting.evaluate(models.baseline, {"lapse": 0.25}, [c])
    np.testing.assert_allclose(ev.per_condition_perf[0], 0.75 * raw + 0.25 / len(c.objects))
    assert ev.per_condition_sse[0] == pytest.approx(((ev.per_condition_perf[0] - c.accuracy) ** 2).sum())


def test_lapse_upper_bound_is_respected(corpus) -> None:
    r = fitting.fit(models.baseline, {}, corpus[:4],
                    budget=fitting.FitBudget(popsize=6, maxiter=20, lapse_max=0.05))
    assert r.params["lapse"] <= 0.05


def test_lapse_name_is_reserved() -> None:
    with pytest.raises(ValueError, match="reserved"):
        parse_parameters({"lapse": "[0, 1]"})
    with pytest.raises(ValueError, match="reserved"):
        fitting.fit(models.baseline, {"lapse": (0.0, 1.0)}, [], fit_lapse=False)


# --- fitting: parameter recovery ---------------------------------------------------

def test_fit_recovers_generating_parameters(corpus) -> None:
    """Synthetic 'human' counts = expected counts under uncfam at known
    parameters (lapse included), so the NLL minimum is exactly there."""
    truth = {"X": 0.2, "B": 3.0, "C": 0.97, "lapse": 0.1}
    synthetic = []
    for c in corpus[:8]:
        m = models.uncfam(truth, c.positional_trials, len(c.words), len(c.objects))
        probs = fitting.choice_probabilities(m, c)
        if c.test is not None:
            continue
        p = (1 - truth["lapse"]) * probs + truth["lapse"] / probs.shape[1]
        n = c.response_matrix.counts.sum(axis=1, keepdims=True)
        synthetic.append(replace(c, response_matrix=replace(c.response_matrix, counts=n * p)))
    assert len(synthetic) >= 6
    r = fitting.fit(models.uncfam, {"X": (0.01, 1.0), "B": (0.1, 10.0)}, synthetic,
                    fixed={"C": truth["C"]},
                    budget=fitting.FitBudget(popsize=8, maxiter=30, objective="multinomial"))
    assert r.params["X"] == pytest.approx(truth["X"], rel=1e-3)
    assert r.params["B"] == pytest.approx(truth["B"], rel=1e-3)
    assert r.params["lapse"] == pytest.approx(truth["lapse"], rel=1e-3)
    assert r.params["C"] == truth["C"]
    assert math.isfinite(r.bic)
