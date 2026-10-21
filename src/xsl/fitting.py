"""Fit XSL theories to the human corpus at the group level.

Objective: a negative log-likelihood of the human test responses in each
condition's `response_matrix`, summed over conditions. Two are available
(`FitBudget.objective`):

  * "binomial" (default, the primary score): every test item is binomial —
    how many people chose the correct referent, out of how many were tested —
    under the theory's predicted accuracy. This scores item-level accuracy,
    weighting each item by its sample size.
  * "multinomial": n-AFC items (no explicit test; every object is a choice)
    are multinomial over all objects, so WHICH wrong object people chose also
    counts; explicit-test items stay binomial. For n-AFC tests 71-80% of this
    NLL is the which-wrong-object part, so fitting it trades item-level
    accuracy for confusion structure (e.g. kachergis fitted this way predicts
    .41 on the hardest conditions, where people score .16). Kept as a
    diagnostic.

  * "subject": the random-ability likelihood of each person's responses.
    Group counts cannot identify differences between people (each person
    answers each item once, so an item's count is binomial however much
    people differ); per-subject response patterns can. Person s has an
    ability offset u_s ~ N(0, ability_sd^2) on the log-odds of the theory's
    predicted accuracy p_i, so P(correct) = (1 - lapse) * logistic(logit p_i
    + u_s) + lapse / n_choices_i; each person's likelihood is the product
    over their items, integrated over u_s by Gauss-Hermite quadrature. The
    spread `ability_sd` is fitted for every theory (a reserved, counted
    parameter, like `lapse`). With ability_sd = 0 it equals the binomial NLL
    up to the binomial coefficients. Needs `Condition.subject_correct`.

Explicit-test items use only the target's count, so they are unaffected by
any mismatch between recorded candidate lists and the objects people chose.

Predicted probabilities are mixed with a guessing rate `lapse` toward
uniform over the choices. `lapse` is FITTED for every theory, as a nuisance
parameter that counts toward k: with a fixed small lapse the likelihood is
dominated by how a theory handles zeros (raw co-occurrence counting, which
gives never-seen pairs probability 0, scores worse than uniform guessing at
lapse = .001 but close to uniform-attention models at its best lapse of ~.3),
not by its learning mechanism. Theories may not declare a parameter named
`lapse`. The bakeoff's n_subj-weighted SSE is reported alongside for
comparison with the xsl-bakeoff fits.

The likelihood treats every choice as independent, which ignores
between-subject variability and makes it overconfident. It is fine for
ranking theories; judge whether two theories truly differ by
cross-validation over conditions (`cross_validate`), not by NLL differences.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution
from scipy.special import gammaln, log_ndtr as norm_logcdf, xlog1py, xlogy

from src.xsl.corpus import Condition
from src.xsl.scoring import Learner, predict_accuracy, predict_referent, sse

LAPSE = "lapse"
LAPSE_BOUNDS = (1e-3, 0.95)
OBJECTIVES = ("binomial", "multinomial", "subject")
DIAGNOSTIC_OBJECTIVES = ("binomial", "multinomial")  # always reported alongside the fitted one
ABILITY_SD = "ability_sd"
ABILITY_SD_BOUNDS = (0.0, 6.0)
RESERVED = (LAPSE, ABILITY_SD)
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)
_FLAT = 14.0   # |logit| beyond which an item's P(correct) is flat to within e^-14
_PANEL = 1.0   # widest Gauss-Legendre panel, in logits
_P_CLIP = 1e-9


def choice_probabilities(m: np.ndarray, condition: Condition, d: float | None = None) -> np.ndarray:
    """Luce choice distribution per item over every object column, shape (n_items, n_objects).

    For explicit-test items the mass sits on the trial's candidates only.
    """
    n_objects = len(condition.objects)
    if condition.test is None:
        M = np.maximum(m, 0.0) ** (1.0 if d is None else d)
        total = M.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, M / total, 1.0 / n_objects)
    out = np.zeros((len(condition.test), n_objects))
    col = {o: j for j, o in enumerate(condition.objects)}
    for i, (word, cands) in enumerate(zip(condition.test.words, condition.test.objects)):
        probs = predict_referent(m, condition.words, condition.objects, word, cands, d)
        for c, p in zip(cands, probs):
            if c in col:
                out[i, col[c]] += p
    return out


def _binomial_nll(k: np.ndarray, n: np.ndarray, p: np.ndarray) -> float:
    log_coef = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    # xlogy / xlog1py: 0 * log(0) = 0, e.g. a single-candidate trial where p = 1
    return float(-(log_coef + xlogy(k, p) + xlog1py(n - k, -p)).sum())


def subject_nll(p_correct: np.ndarray, condition: Condition, lapse: float, ability_sd: float) -> float:
    """Random-ability NLL of `condition.subject_correct` given per-item accuracy `p_correct` (before guessing).

    Each person's likelihood is integrated over their ability u ~ N(0,
    ability_sd^2). Every item's P(correct) depends on u only through
    logistic(logit p_i + u), which is flat (to within e^-14) once |logit p_i
    + u| > 14. So outside the window [-14 - max logit p, 14 - min logit p]
    every person's likelihood is constant - its all-guessing or all-knowing
    limit - and those tails integrate exactly against the normal CDF. Inside
    the window (clipped to +-z ability_sd, z large enough that the ignored
    prior mass is negligible next to every person's likelihood) the integrand is resolved by
    composite Gauss-Legendre on panels at most 1 logit and 1 ability_sd
    wide, shared by everyone in the condition. (Gauss-Hermite nodes, fixed or adapted to each
    person's posterior peak, failed: with guessing the likelihood never
    reaches 0 or 1, so a high-accuracy person's posterior is a Gaussian cut
    by a step, not a peak.)
    """
    y = condition.subject_correct
    if y is None:
        raise ValueError(f"{condition.label} has no per-subject responses (export subject_responses.json)")
    k = n_choices_per_item(condition)
    p_clip = np.clip(p_correct, _P_CLIP, 1 - _P_CLIP)
    eta = np.log(p_clip) - np.log1p(-p_clip)                              # (I,)
    seen = ~np.isnan(y)
    yy = np.where(seen, y, 0.0)                                           # (S, I)
    wrong = (seen & (yy == 0)).astype(float)

    def log_lik(pc):  # pc: (Q, I) -> (S, Q)
        pc = np.clip(pc, _P_CLIP, 1 - _P_CLIP)
        return yy @ np.log(pc).T + wrong @ np.log1p(-pc).T

    if ability_sd < 1e-8:
        return float(-log_lik(((1 - lapse) * p_clip + lapse / k)[None, :])[:, 0].sum())

    # Prior mass beyond +-z ability_sd is ignored; z is chosen so that mass is
    # negligible even against the person with the smallest likelihood at
    # ability_sd = 0 (likelihoods can be ~e^-200 with a small guessing rate).
    worst = float(-log_lik(((1 - lapse) * p_clip + lapse / k)[None, :])[:, 0].max())
    z = np.sqrt(2.0 * (worst + 40.0))
    lo, hi = -z * ability_sd, z * ability_sd
    # outside [w1, w2] every item's P(correct) is at its limit (flat to within e^-14)
    w1, w2 = -_FLAT - eta.max(), _FLAT - eta.min()
    a, b = max(lo, w1), min(hi, w2)
    parts = []
    if b > a:
        # panels at most 1 logit wide (item transitions) and 1 ability_sd wide (the prior)
        n_panels = int(np.ceil((b - a) / min(_PANEL, ability_sd)))
        edges = np.linspace(a, b, n_panels + 1)
        half = (edges[1:] - edges[:-1]) / 2.0
        nodes = ((edges[1:] + edges[:-1]) / 2.0)[:, None] + half[:, None] * _GL_NODES[None, :]
        nodes, log_w = nodes.ravel(), (np.log(half)[:, None] + np.log(_GL_WEIGHTS)[None, :]).ravel()
        q = 1.0 / (1.0 + np.exp(-(eta[None, :] + nodes[:, None])))        # (Q, I)
        log_prior = -0.5 * (nodes / ability_sd) ** 2 - np.log(ability_sd * np.sqrt(2 * np.pi))
        parts.append(log_lik((1 - lapse) * q + lapse / k[None, :]) + (log_prior + log_w)[None, :])
    # flat tails, exactly: the likelihood at its u -> -inf / +inf limit times the
    # prior mass beyond the window (zero if the window reaches past +-z ability_sd)
    if w1 > lo:
        parts.append(log_lik((lapse / k)[None, :]) + norm_logcdf(w1 / ability_sd))
    if w2 < hi:
        parts.append(log_lik((1 - lapse + lapse / k)[None, :]) + norm_logcdf(-w2 / ability_sd))
    log_L = np.logaddexp.reduce(np.concatenate(parts, axis=1), axis=1)
    return float(-log_L.sum())


def condition_nll(m: np.ndarray, condition: Condition, d: float | None = None, lapse: float = 0.0,
                  objective: str = "binomial", ability_sd: float = 0.0) -> float:
    """Negative log-likelihood of `condition`'s human test responses under association matrix `m`."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if objective == "subject":
        return subject_nll(predict_accuracy(m, condition, d=d), condition, lapse, ability_sd)
    rm = condition.response_matrix
    if rm is None:
        raise ValueError(f"{condition.label} has no response matrix")
    probs = choice_probabilities(m, condition, d)
    counts = rm.counts
    n = counts.sum(axis=1)
    if condition.test is None:
        p = (1 - lapse) * probs + lapse / probs.shape[1]
        if objective == "multinomial":
            log_coef = gammaln(n + 1) - gammaln(counts + 1).sum(axis=1)
            return float(-(log_coef + xlogy(counts, p).sum(axis=1)).sum())
        # word i's correct referent is object i (sorted order), as in get_perf
        items = np.arange(len(condition.words))
        return _binomial_nll(counts[items, items], n, p[items, items])
    col = {o: j for j, o in enumerate(condition.objects)}
    targets = condition.test.correct or condition.test.words
    j = np.array([col[t] for t in targets])
    items = np.arange(len(targets))
    k_cands = np.array([float(len(c)) for c in condition.test.objects])
    p = (1 - lapse) * probs[items, j] + lapse / k_cands
    return _binomial_nll(counts[items, j], n, p)


@dataclass
class Evaluation:
    """A theory scored on a set of conditions at one parameter setting."""

    params: dict[str, float]
    nll: float
    weighted_sse: float
    per_condition_nll: list[float]
    per_condition_sse: list[float]
    per_condition_perf: list[np.ndarray]
    objective: str = "binomial"
    other_nll: dict[str, float] = field(default_factory=dict)  # total NLL under the other objective(s)


def total_choices(conditions: Sequence[Condition]) -> int:
    return int(sum(c.response_matrix.counts.sum() for c in conditions))


def n_choices_per_item(condition: Condition) -> np.ndarray:
    """How many options each test item offers (all objects, or the m-AFC candidates)."""
    if condition.test is None:
        return np.full(len(condition.words), float(len(condition.objects)))
    return np.array([float(len(c)) for c in condition.test.objects])


def evaluate(learner: Learner, params: dict[str, float], conditions: Sequence[Condition],
             objective: str = "binomial") -> Evaluation:
    """NLL under `objective` (and, for reference, the other objectives), plus
    per-item accuracy and SSE with the guessing rate mixed in
    ((1 - lapse) * p + lapse / n_choices), i.e. the accuracy the likelihood scores."""
    nlls, sses, perfs = [], [], []
    others = {o: 0.0 for o in DIAGNOSTIC_OBJECTIVES if o != objective}
    d = params.get("ch_dec")
    lapse = params.get(LAPSE, 0.0)
    ability_sd = params.get(ABILITY_SD, 0.0)
    for c in conditions:
        m = learner(params, c.positional_trials, len(c.words), len(c.objects))
        perf = (1 - lapse) * predict_accuracy(m, c, d=d) + lapse / n_choices_per_item(c)
        perfs.append(perf)
        nlls.append(condition_nll(m, c, d, lapse, objective, ability_sd))
        for o in others:
            others[o] += condition_nll(m, c, d, lapse, o)
        sses.append(sse(perf, c))
    weights = np.array([c.n_subj for c in conditions], dtype=float)
    return Evaluation(
        params=dict(params),
        nll=float(sum(nlls)),
        weighted_sse=float((np.array(sses) * weights).sum() / weights.sum()),
        per_condition_nll=nlls,
        per_condition_sse=sses,
        per_condition_perf=perfs,
        objective=objective,
        other_nll=others,
    )


_GH40_NODES, _GH40_WEIGHTS = np.polynomial.hermite.hermgauss(40)


def population_accuracy(learner: Learner, params: dict[str, float], condition: Condition) -> np.ndarray:
    """Predicted mean accuracy per item across people.

    Under the random-ability objective a person's accuracy on item i is
    (1 - lapse) * logistic(logit p_i + u) + lapse / n_choices_i with u ~ N(0,
    ability_sd^2), so the population mean is that averaged over u (40-node
    Gauss-Hermite; the integrand is smooth). Without `ability_sd` this is the
    usual lapse-mixed accuracy.
    """
    m = learner(params, condition.positional_trials, len(condition.words), len(condition.objects))
    p = predict_accuracy(m, condition, d=params.get("ch_dec"))
    lapse = params.get(LAPSE, 0.0)
    k = n_choices_per_item(condition)
    sd = params.get(ABILITY_SD, 0.0)
    if sd <= 0:
        return (1 - lapse) * p + lapse / k
    pc = np.clip(p, _P_CLIP, 1 - _P_CLIP)
    eta = np.log(pc) - np.log1p(-pc)
    u = np.sqrt(2.0) * sd * _GH40_NODES
    mean_q = (_GH40_WEIGHTS[:, None] / np.sqrt(np.pi) / (1.0 + np.exp(-(eta[None, :] + u[:, None])))).sum(axis=0)
    return (1 - lapse) * mean_q + lapse / k


@dataclass
class FitResult:
    params: dict[str, float]
    evaluation: Evaluation
    n_free: int
    n_choices: int
    n_evals: int
    seconds: float
    at_bound: list[str] = field(default_factory=list)

    @property
    def nll(self) -> float:
        return self.evaluation.nll

    @property
    def bic(self) -> float:
        return 2 * self.nll + self.n_free * math.log(self.n_choices)

    @property
    def aic(self) -> float:
        return 2 * self.nll + 2 * self.n_free


@dataclass(frozen=True)
class FitBudget:
    """Fitting settings: differential_evolution's budget (defaults sized for
    the inner critique loop; raise them for final fits), the upper bound of
    the fitted guessing rate (`lapse_max`, lowered for sensitivity checks),
    and the likelihood being minimised (`objective`, see module docstring).
    Kept in one object because it travels to every fit, including CV worker
    processes."""

    popsize: int = 10
    maxiter: int = 40
    tol: float = 1e-4
    polish: bool = True
    seed: int = 0
    lapse_max: float = LAPSE_BOUNDS[1]
    objective: str = "binomial"
    restarts: int = 1   # independent DE runs (seeds seed, seed+1, ...); the best is kept
    workers: int = 1    # processes evaluating each DE population (needs a theory, see fit)

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")


class _NLLObjective:
    """Summed NLL of `conditions` at a parameter vector.

    A picklable callable so differential_evolution can score a population in
    worker processes: exec'd learners cannot be pickled, so a worker rebuilds
    the learner from the theory's source (`theory_dump`) on first use.
    Failures and non-finite totals score +inf.
    """

    def __init__(self, conditions, names, fixed, objective, *, learner=None, theory_dump=None):
        self.conditions = list(conditions)
        self.names = list(names)
        self.fixed = dict(fixed)
        self.objective = objective
        self.theory_dump = theory_dump
        self._learner = learner

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_learner"] = None
        return state

    def learner(self) -> Learner:
        if self._learner is None:
            from src.xsl.theory import XSLTheory

            self._learner = XSLTheory(**self.theory_dump).learner()
        return self._learner

    def __call__(self, x) -> float:
        params = {**self.fixed, **dict(zip(self.names, map(float, x)))}
        learner = self.learner()
        d = params.get("ch_dec")
        lapse = params.get(LAPSE, 0.0)
        ability_sd = params.get(ABILITY_SD, 0.0)
        total = 0.0
        try:
            with np.errstate(all="ignore"):
                for c in self.conditions:
                    m = learner(params, c.positional_trials, len(c.words), len(c.objects))
                    total += condition_nll(m, c, d, lapse, self.objective, ability_sd)
        except (ArithmeticError, ValueError, FloatingPointError):
            return float("inf")
        return total if math.isfinite(total) else float("inf")


def fit(
    learner: Learner,
    free: dict[str, tuple[float, float]],
    conditions: Sequence[Condition],
    *,
    fixed: dict[str, float] | None = None,
    budget: FitBudget = FitBudget(),
    fit_lapse: bool = True,
    x0: dict[str, float] | None = None,
    theory_dump: dict | None = None,
) -> FitResult:
    """Minimise the summed NLL over `free` parameters with differential evolution.

    With `fit_lapse` (the default) the guessing rate `lapse` is fitted too and
    counts as a free parameter; otherwise it is taken from `fixed` (default 0).

    `budget.restarts` independent runs are made and the best is kept. `x0`
    (e.g. a previous fit) is clipped into the bounds and seeded into the first
    run's population, so that run never ends worse than `x0`. Parallel
    population evaluation (`budget.workers > 1`) needs `theory_dump`, the
    theory's `model_dump()`, so workers can rebuild the learner.
    """
    fixed = dict(fixed or {})
    free = dict(free)
    if LAPSE in free or (fit_lapse and LAPSE in fixed):
        raise ValueError(f"{LAPSE!r} is reserved for the fitted guessing rate")
    if ABILITY_SD in free:
        raise ValueError(f"{ABILITY_SD!r} is reserved for the fitted between-subject spread")
    if fit_lapse:
        free[LAPSE] = (LAPSE_BOUNDS[0], budget.lapse_max)
    if budget.objective == "subject" and ABILITY_SD not in fixed:
        free[ABILITY_SD] = ABILITY_SD_BOUNDS
    names = list(free)
    bounds = [free[n] for n in names]
    t0 = time.perf_counter()
    objective = _NLLObjective(conditions, names, fixed, budget.objective,
                              learner=learner, theory_dump=theory_dump)

    if not names:
        objective(np.array([]))
        best, n_evals = {}, 1
    else:
        start = None
        if x0 is not None:
            # clip a hair inside: scipy rescales x0 to [0, 1] and rejects a
            # value sitting exactly on a bound once rounding makes it -1e-17
            start = np.array([min(max(float(x0.get(n, (lo + hi) / 2)), lo + 1e-12 * (hi - lo)),
                                  hi - 1e-12 * (hi - lo))
                              for n, (lo, hi) in zip(names, bounds)])
        pool = None
        if budget.workers > 1:
            if theory_dump is None:
                raise ValueError("parallel fitting (budget.workers > 1) needs theory_dump")
            import multiprocessing

            pool = multiprocessing.get_context("spawn").Pool(budget.workers)
        try:
            best_res, n_evals = None, 0
            for r in range(max(1, budget.restarts)):
                kwargs = dict(popsize=budget.popsize, maxiter=budget.maxiter, tol=budget.tol,
                              polish=budget.polish, seed=budget.seed + r, init="sobol")
                if r == 0 and start is not None:
                    kwargs["x0"] = start
                if pool is not None:
                    kwargs.update(workers=pool.map, updating="deferred")
                with np.errstate(all="ignore"):
                    res = differential_evolution(objective, bounds=bounds, **kwargs)
                n_evals += int(res.nfev)
                if best_res is None or res.fun < best_res.fun:
                    best_res = res
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        best = {n: float(v) for n, v in zip(names, best_res.x)}
    params = {**fixed, **best}
    at_bound = [
        n for n, v in best.items()
        if math.isclose(v, free[n][0], rel_tol=1e-3, abs_tol=1e-6)
        or math.isclose(v, free[n][1], rel_tol=1e-3, abs_tol=1e-6)
    ]
    return FitResult(
        params=params,
        evaluation=evaluate(learner, params, conditions, budget.objective),
        n_free=len(names),
        n_choices=total_choices(conditions),
        n_evals=n_evals,
        seconds=time.perf_counter() - t0,
        at_bound=at_bound,
    )


@dataclass
class CVResult:
    """k-fold cross-validation over conditions."""

    heldout_nll: float                      # summed over all held-out conditions
    heldout_weighted_sse: float
    per_condition_nll: list[float]          # in corpus order, each from the fold that held it out
    fold_params: list[dict[str, float]]


def cv_folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    """The k folds of condition indices. Same (n, k, seed) -> same folds, so
    every theory is cross-validated on an identical split. At most n folds."""
    return np.array_split(np.random.default_rng(seed).permutation(n), min(k, n))


def _fit_fold(learner: Learner, free, fixed, conditions, held, budget, fit_lapse, x0=None, theory_dump=None):
    held_set = set(held.tolist())
    train = [c for i, c in enumerate(conditions) if i not in held_set]
    result = fit(learner, free, train, fixed=fixed, budget=budget, fit_lapse=fit_lapse,
                 x0=x0, theory_dump=theory_dump)
    ev = evaluate(learner, result.params, [conditions[i] for i in held], budget.objective)
    return held, result.params, ev.per_condition_nll, ev.per_condition_sse


def _assemble_cv(conditions, fold_results) -> CVResult:
    n = len(conditions)
    per_nll = [math.nan] * n
    per_sse = [math.nan] * n
    fold_params = []
    for held, params, nlls, sses in fold_results:
        fold_params.append(params)
        for i, nll_i, sse_i in zip(held, nlls, sses):
            per_nll[i] = nll_i
            per_sse[i] = sse_i
    weights = np.array([c.n_subj for c in conditions], dtype=float)
    return CVResult(
        heldout_nll=float(sum(per_nll)),
        heldout_weighted_sse=float((np.array(per_sse) * weights).sum() / weights.sum()),
        per_condition_nll=per_nll,
        fold_params=fold_params,
    )


def cross_validate(
    learner: Learner,
    free: dict[str, tuple[float, float]],
    conditions: Sequence[Condition],
    *,
    fixed: dict[str, float] | None = None,
    k: int = 5,
    seed: int = 0,
    budget: FitBudget = FitBudget(),
    fit_lapse: bool = True,
) -> CVResult:
    """Fit on k-1 folds of conditions, score the held-out fold; repeat for every fold."""
    conditions = list(conditions)
    return _assemble_cv(conditions, [
        _fit_fold(learner, free, fixed, conditions, held, budget, fit_lapse)
        for held in cv_folds(len(conditions), k, seed)
    ])


def _theory_fold(theory_dump: dict, conditions, held, budget, x0=None):
    # runs in a worker process: rebuild the theory from its source there,
    # since exec'd learners cannot be pickled
    from src.xsl.theory import XSLTheory

    theory = XSLTheory(**theory_dump)
    return _fit_fold(theory.learner(), theory.free_parameters, theory.fixed_parameters,
                     conditions, held, budget, True, x0)


def cross_validate_theory(
    theory,
    conditions: Sequence[Condition],
    *,
    k: int = 5,
    seed: int = 0,
    budget: FitBudget = FitBudget(),
    workers: int = 1,
    x0: dict[str, float] | None = None,
) -> CVResult:
    """`cross_validate` for an `XSLTheory`.

    Parallelism is either across folds (`workers` processes, one fold each)
    or, when `budget.workers > 1`, within each fit (folds then run one after
    another, each fit scoring its populations in `budget.workers` processes).
    `x0` (e.g. the full-corpus fit) warm-starts every fold.
    """
    conditions = list(conditions)
    folds = cv_folds(len(conditions), k, seed)
    dump = theory.model_dump()
    if workers <= 1 or budget.workers > 1:
        return _assemble_cv(conditions, [
            _fit_fold(theory.learner(), theory.free_parameters, theory.fixed_parameters,
                      conditions, held, budget, True, x0, dump)
            for held in folds
        ])
    from concurrent.futures import ProcessPoolExecutor

    n_folds = len(folds)
    with ProcessPoolExecutor(max_workers=min(workers, n_folds)) as pool:
        results = list(pool.map(_theory_fold, [dump] * n_folds, [conditions] * n_folds, folds,
                                [budget] * n_folds, [x0] * n_folds))
    return _assemble_cv(conditions, results)


def fit_theory(theory, conditions: Sequence[Condition], *, budget: FitBudget = FitBudget(),
               x0: dict[str, float] | None = None) -> FitResult:
    """`fit` for an `XSLTheory`."""
    return fit(theory.learner(), theory.free_parameters, conditions,
               fixed=theory.fixed_parameters, budget=budget, x0=x0, theory_dump=theory.model_dump())
