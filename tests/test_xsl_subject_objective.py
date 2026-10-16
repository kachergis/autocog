"""The per-subject random-ability likelihood (objective "subject")."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import integrate
from scipy.special import expit, gammaln, logit
from scipy.stats import norm

from src.xsl import fitting, models
from src.xsl.corpus import Condition, load_corpus
from src.xsl.scoring import predict_accuracy
from src.xsl.theory import parse_parameters

PARAMS = {"X": 0.3, "B": 1.5, "C": 0.97}


@pytest.fixture(scope="module")
def corpus() -> list[Condition]:
    return load_corpus()


def _log_binom_coef(c: Condition) -> float:
    n = c.response_matrix.counts.sum(axis=1)
    if c.test is None:
        k = c.response_matrix.counts[np.arange(len(c.words)), np.arange(len(c.words))]
    else:
        col = {o: j for j, o in enumerate(c.objects)}
        targets = c.test.correct or c.test.words
        k = np.array([c.response_matrix.counts[i, col[t]] for i, t in enumerate(targets)])
    return float((gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)).sum())


def test_every_condition_has_subject_responses(corpus) -> None:
    assert all(c.subject_correct is not None for c in corpus)
    assert sum(c.subject_correct.shape[0] for c in corpus) == 1609
    for c in corpus:
        assert c.subject_correct.shape == (c.n_subj, c.n_items)
        np.testing.assert_allclose(np.nanmean(c.subject_correct, axis=0), c.accuracy, atol=1e-12)


def test_zero_spread_is_the_binomial_up_to_coefficients(corpus) -> None:
    for c in corpus[:6] + [x for x in corpus if x.test is not None][:2]:
        m = models.uncfam(PARAMS, c.positional_trials, len(c.words), len(c.objects))
        b = fitting.condition_nll(m, c, lapse=0.2, objective="binomial")
        s = fitting.condition_nll(m, c, lapse=0.2, objective="subject", ability_sd=0.0)
        assert s == pytest.approx(b + _log_binom_coef(c), rel=1e-10)


def test_quadrature_matches_direct_integration(corpus) -> None:
    c = corpus[0]
    m = models.uncfam(PARAMS, c.positional_trials, len(c.words), len(c.objects))
    p = predict_accuracy(m, c)
    lapse, sd, k = 0.15, 1.3, len(c.objects)
    y = c.subject_correct[:3]  # three people
    expected = 0.0
    for row in y:
        def integrand(u, row=row):
            pc = (1 - lapse) * expit(logit(np.clip(p, 1e-9, 1 - 1e-9)) + u) + lapse / k
            return np.exp((row * np.log(pc) + (1 - row) * np.log1p(-pc)).sum()) * norm.pdf(u, scale=sd)
        expected -= math.log(integrate.quad(integrand, -12 * sd, 12 * sd, limit=200)[0])
    one = replace(c, subject_correct=y)
    assert fitting.subject_nll(p, one, lapse, sd) == pytest.approx(expected, rel=1e-7)


def test_untested_items_are_skipped(corpus) -> None:
    c = corpus[0]
    m = models.uncfam(PARAMS, c.positional_trials, len(c.words), len(c.objects))
    p = predict_accuracy(m, c)
    lapse, k = 0.1, len(c.objects)
    y = c.subject_correct[:2].copy()
    y[0, :5] = np.nan  # person 0 was not tested on the first five items

    def person_nll(row, pp):
        def f(u):
            pc = (1 - lapse) * expit(logit(np.clip(pp, 1e-9, 1 - 1e-9)) + u) + lapse / k
            return np.exp((row * np.log(pc) + (1 - row) * np.log1p(-pc)).sum()) * norm.pdf(u)
        return -math.log(integrate.quad(f, -12, 12, limit=200)[0])

    expected = person_nll(y[0, 5:], p[5:]) + person_nll(y[1], p)
    assert fitting.subject_nll(p, replace(c, subject_correct=y), lapse, 1.0) == pytest.approx(expected, rel=1e-7)


def test_spread_and_guessing_are_recovered(corpus) -> None:
    """Simulate people with known ability spread and guessing rate on the corpus
    designs, then fit those two with the learner's parameters held at truth."""
    rng = np.random.default_rng(1)
    true_sd, true_lapse, n_people = 1.4, 0.12, 300
    synthetic = []
    for c in [x for x in corpus if x.test is None][:8]:
        m = models.uncfam(PARAMS, c.positional_trials, len(c.words), len(c.objects))
        eta = logit(np.clip(predict_accuracy(m, c), 1e-9, 1 - 1e-9))
        u = rng.normal(0, true_sd, size=(n_people, 1))
        pc = (1 - true_lapse) * expit(eta[None, :] + u) + true_lapse / len(c.objects)
        y = (rng.random(pc.shape) < pc).astype(float)
        synthetic.append(replace(c, subject_correct=y))
    r = fitting.fit(models.uncfam, {}, synthetic, fixed=PARAMS,
                    budget=fitting.FitBudget(popsize=8, maxiter=40, objective="subject"))
    assert set(r.params) == {*PARAMS, "lapse", "ability_sd"}
    assert r.n_free == 2
    assert r.params["ability_sd"] == pytest.approx(true_sd, abs=0.15)
    assert r.params["lapse"] == pytest.approx(true_lapse, abs=0.04)


def test_ability_sd_is_reserved() -> None:
    with pytest.raises(ValueError, match="reserved"):
        parse_parameters({"ability_sd": "[0, 1]"})
    with pytest.raises(ValueError, match="reserved"):
        fitting.fit(models.baseline, {"ability_sd": (0.0, 1.0)}, [])


def test_fit_adds_ability_sd_only_for_the_subject_objective(corpus) -> None:
    tiny = dict(popsize=3, maxiter=2, polish=False)
    b = fitting.fit(models.baseline, {}, corpus[:3], budget=fitting.FitBudget(**tiny))
    s = fitting.fit(models.baseline, {}, corpus[:3], budget=fitting.FitBudget(objective="subject", **tiny))
    assert "ability_sd" not in b.params and b.n_free == 1
    assert "ability_sd" in s.params and s.n_free == 2
    assert s.evaluation.objective == "subject"
    assert set(s.evaluation.other_nll) == {"binomial", "multinomial"}


@pytest.mark.parametrize("label,lapse,sd", [
    ("215", 0.4, 6.0),    # high-accuracy people: posterior is a Gaussian cut by a step
    ("215", 0.02, 6.0),
    ("211", 0.4, 0.05),   # tiny spread: panels must resolve the narrow prior
    ("301", 0.0, 0.01),   # no guessing: flat tails start at the item window, not at the prior cut-off
    ("201", 0.15, 2.5),
])
def test_integration_matches_a_dense_grid(corpus, label, lapse, sd) -> None:
    """Regression cases that broke earlier quadrature schemes."""
    from scipy.special import logsumexp

    c = next(x for x in corpus if x.label == label)
    m = models.uncfam(PARAMS, c.positional_trials, len(c.words), len(c.objects))
    p = predict_accuracy(m, c)
    k = fitting.n_choices_per_item(c)
    eta = logit(np.clip(p, 1e-9, 1 - 1e-9))
    span = 25 * sd + 40
    u = np.linspace(-span, span, 400001)
    pc = np.clip((1 - lapse) * expit(eta[None, :] + u[:, None]) + lapse / k[None, :], 1e-9, 1 - 1e-9)
    lp, l1, prior = np.log(pc), np.log1p(-pc), norm.logpdf(u, scale=sd)
    grid = -sum(logsumexp((row * lp + (1 - row) * l1).sum(1) + prior) + math.log(u[1] - u[0])
                for row in c.subject_correct)
    assert fitting.subject_nll(p, c, lapse, sd) == pytest.approx(grid, abs=1e-5 * c.n_subj)
