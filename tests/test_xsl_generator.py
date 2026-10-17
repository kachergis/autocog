"""XSL theory generator: accept gate, retries and the critic, with a fake LLM.

The fake client stands in for `src.llm` clients (same `.chat` contract), so
these tests need no LLM SDK or API key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

import main_xsl_offline
from src.xsl.corpus import load_corpus
from src.xsl.fitting import FitBudget, cross_validate_theory, fit_theory
from src.xsl.generator import XSLTheoryGenerator
from src.xsl.prompts import Entry
from src.xsl.theory import XSLTheory

BUDGET = FitBudget(popsize=4, maxiter=5, polish=False)

# chance everywhere, with one useless free parameter: a poor fit
UNIFORM = {
    "name": "Uniform guessing",
    "theory": "Nothing is learned.",
    "learn": "def learn(params, trials, n_words, n_objects):\n    return np.ones((n_words, n_objects))\n",
    "parameters": {"k": "[0, 1]"},
}
# raw co-occurrence counting, zero parameters: a much better fit than UNIFORM
COUNTING = {
    "name": "Counting",
    "theory": "Count co-occurrences.",
    "learn": (
        "def learn(params, trials, n_words, n_objects):\n"
        "    m = np.zeros((n_words, n_objects))\n"
        "    for w, o in trials:\n"
        "        m[np.ix_(w, o)] += 1\n"
        "    return m\n"
    ),
    "parameters": {},
}


def critique(verdict: str) -> str:
    return json.dumps({"interpretation": "i", "verdict": verdict, "rationale": f"r-{verdict}"})


@dataclass
class _Result:
    text: str
    usage: dict = field(default_factory=dict)


class FakeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def chat(self, messages, system, response_schema=None):
        self.prompts.append(messages[-1]["content"])
        return _Result(text=self.responses.pop(0))


@pytest.fixture(scope="module")
def conditions():
    return load_corpus()[:4]


@pytest.fixture(scope="module")
def leaderboard(conditions):
    theory = XSLTheory.from_yaml("theories/xsl/baseline.yaml")
    return [Entry(label="baseline", theory=theory, fit=fit_theory(theory, conditions, budget=BUDGET),
                  cv=cross_validate_theory(theory, conditions, k=2, budget=BUDGET))]


def _generator(conditions, responses):
    client = FakeClient(responses)
    return XSLTheoryGenerator(llm_client=client, conditions=conditions, budget=BUDGET, cv_folds=2), client


def _gate_is_cv(history) -> None:
    """Each accepted candidate beat every earlier accepted one on held-out NLL."""
    best = float("inf")
    for it in history:
        if it.candidate is None:
            continue
        assert it.cv is not None
        assert it.accepted == (it.score < best)
        best = min(best, it.score) if it.accepted else best


def test_better_candidate_is_accepted_and_continue_stops(conditions, leaderboard, tmp_path) -> None:
    gen, client = _generator(conditions, [
        json.dumps(UNIFORM), critique("regenerate"),
        json.dumps(COUNTING), critique("continue"),
    ])
    result = gen.propose_theory(guide="g", leaderboard=leaderboard, max_iters=5, workspace=tmp_path)
    assert [it.accepted for it in result.history] == [True, True]
    assert result.best.candidate.name == "Counting"
    _gate_is_cv(result.history)
    assert not client.responses  # stopped after the accepted 'continue'
    # the second proposal was shown the first candidate as the running best
    assert "CURRENT BEST CANDIDATE" in client.prompts[2] and "Uniform guessing" in client.prompts[2]
    assert (tmp_path / "prompts" / "generation_iter_01_attempt_00.md").exists()
    assert (tmp_path / "prompts" / "critique_iter_01.md").exists()


def test_worse_candidate_is_rejected_and_continue_is_ignored(conditions, leaderboard) -> None:
    gen, client = _generator(conditions, [
        json.dumps(COUNTING), critique("regenerate"),
        json.dumps(UNIFORM), critique("continue"),   # rejected, so 'continue' must not stop the loop
        json.dumps(UNIFORM), critique("regenerate"),
    ])
    result = gen.propose_theory(guide="g", leaderboard=leaderboard, max_iters=3)
    assert [it.accepted for it in result.history] == [True, False, False]
    assert result.best.candidate.name == "Counting"
    _gate_is_cv(result.history)
    assert not client.responses


def test_unparseable_or_broken_candidates_are_retried(conditions, leaderboard) -> None:
    broken = {**COUNTING, "learn": "def learn(params, trials, n_words, n_objects):\n    return -1\n"}
    gen, client = _generator(conditions, [
        "not json at all", json.dumps(broken), json.dumps(COUNTING), critique("continue"),
    ])
    result = gen.propose_theory(guide="g", leaderboard=leaderboard, max_iters=2)
    assert result.best.candidate.name == "Counting"
    # each retry tells the model what went wrong
    assert "YOUR PREVIOUS ATTEMPT FAILED" in client.prompts[1]
    assert "YOUR PREVIOUS ATTEMPT FAILED" in client.prompts[2]


def test_no_valid_candidate_raises(conditions, leaderboard) -> None:
    gen, _ = _generator(conditions, ["nope"] * 3)
    with pytest.raises(RuntimeError, match="no valid theory"):
        gen.propose_theory(guide="g", leaderboard=leaderboard, max_iters=1)


def test_leaderboard_persistence_round_trip(conditions, leaderboard, tmp_path) -> None:
    main_xsl_offline.save(tmp_path, leaderboard)
    loaded = main_xsl_offline.load(tmp_path, conditions)
    assert [e.label for e in loaded] == ["baseline"]
    assert loaded[0].fit.nll == pytest.approx(leaderboard[0].fit.nll, rel=1e-12)
    assert loaded[0].cv.per_condition_nll == leaderboard[0].cv.per_condition_nll
    assert loaded[0].score == leaderboard[0].score
    assert "baseline" in (tmp_path / "cv.md").read_text()
    assert (tmp_path / "theories" / "baseline.yaml").exists()
    assert "baseline" in (tmp_path / "leaderboard.md").read_text()


def test_openrouter_client_for_closed_models(monkeypatch) -> None:
    pytest.importorskip("anthropic")  # src.llm needs every SDK installed
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    args = main_xsl_offline.parse_args(["--llm_provider", "openrouter", "--llm_model", "anthropic/claude-opus-5.5"])
    client = main_xsl_offline.make_llm_client(args)
    assert client.model == "anthropic/claude-opus-5.5"
    assert "quantizations" not in client.extra_body["provider"]
    assert client.extra_body["provider"]["require_parameters"] is False
    assert client.extra_body["reasoning"]["effort"] == "high"
    args = main_xsl_offline.parse_args(["--llm_provider", "openrouter", "--llm_model", "deepseek/deepseek-v4-pro-0813"])
    open_weight = main_xsl_offline.make_llm_client(args).extra_body["provider"]
    assert open_weight["quantizations"] == ["fp8"] and open_weight["require_parameters"] is True


def test_legacy_cv_json_is_picked_up(conditions, leaderboard, tmp_path) -> None:
    # runs before per-entry CV kept it only in cv.json, keyed by condition label
    cv = leaderboard[0].cv
    main_xsl_offline.save(tmp_path, [Entry(label="baseline", theory=leaderboard[0].theory, fit=leaderboard[0].fit)])
    (tmp_path / "cv.json").write_text(json.dumps([{
        "label": "baseline", "heldout_nll": cv.heldout_nll, "heldout_weighted_sse": cv.heldout_weighted_sse,
        "fold_params": cv.fold_params,
        "per_condition_nll": dict(zip((c.label for c in conditions), cv.per_condition_nll)),
    }]))
    loaded = main_xsl_offline.load(tmp_path, conditions)
    assert loaded[0].cv.per_condition_nll == cv.per_condition_nll


def test_leaderboard_ranks_by_held_out_nll(conditions, leaderboard) -> None:
    from dataclasses import replace

    from src.xsl.prompts import leaderboard_table

    base = leaderboard[0]
    better_cv = replace(base.cv, heldout_nll=base.cv.heldout_nll - 1)
    worse_bic = replace(base.fit, n_free=base.fit.n_free + 5)  # higher BIC, better CV
    table = leaderboard_table([base, Entry(label="cv_winner", theory=base.theory, fit=worse_bic, cv=better_cv)])
    rows = [line for line in table.splitlines() if line.startswith("| 1 ") or line.startswith("| 2 ")]
    assert "cv_winner" in rows[0] and "baseline" in rows[1]


def test_add_theories_skips_duplicate_labels(conditions) -> None:
    theory = XSLTheory.from_yaml("theories/xsl/baseline.yaml")
    entries: list = []
    main_xsl_offline.add_theories(entries, [("baseline", theory), ("baseline", theory)], conditions, BUDGET)
    assert [e.label for e in entries] == ["baseline"]


def test_loading_under_a_different_objective_fails(conditions, leaderboard, tmp_path) -> None:
    main_xsl_offline.save(tmp_path, leaderboard)  # fitted with the default (binomial) objective
    with pytest.raises(ValueError, match="binomial objective"):
        main_xsl_offline.load(tmp_path, conditions, "multinomial")
    # runs from before the objective was recorded were multinomial
    rows = json.loads((tmp_path / "leaderboard.json").read_text())
    for r in rows:
        del r["objective"]
    (tmp_path / "leaderboard.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="multinomial objective"):
        main_xsl_offline.load(tmp_path, conditions, "binomial")


def test_prompt_describes_the_objective() -> None:
    from src.xsl.prompts import domain_text

    assert "ACCURACY" in domain_text(100, "binomial")
    assert "WHICH wrong object" in domain_text(100, "multinomial")


def test_warm_start_takes_shared_names_in_priority_order() -> None:
    from src.xsl.generator import warm_start

    theory = XSLTheory(theory="t", learn=COUNTING["learn"], parameters={"a": "[0, 1]", "z": "[0, 9]"})
    x0 = warm_start(theory, [{"a": 0.2, "lapse": 0.01}, {"a": 0.9, "z": 4.0, "ability_sd": 1.5}])
    assert x0 == {"a": 0.2, "z": 4.0, "lapse": 0.01, "ability_sd": 1.5}
    assert warm_start(theory, []) == {}


def test_generator_runs_under_the_subject_objective(conditions) -> None:
    budget = FitBudget(popsize=3, maxiter=3, polish=False, objective="subject")
    base = XSLTheory.from_yaml("theories/xsl/baseline.yaml")
    board = [Entry(label="baseline", theory=base, fit=fit_theory(base, conditions, budget=budget),
                   cv=cross_validate_theory(base, conditions, k=2, budget=budget))]
    client = FakeClient([json.dumps(COUNTING), critique("continue")])
    gen = XSLTheoryGenerator(llm_client=client, conditions=conditions, budget=budget, cv_folds=2)
    result = gen.propose_theory(guide="g", leaderboard=board, max_iters=1)
    assert "ability_sd" in result.best.fit.params
    assert result.best.cv is not None and result.best.fit.evaluation.objective == "subject"
