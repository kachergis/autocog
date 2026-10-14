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
from src.xsl.fitting import FitBudget, fit_theory
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
    return [Entry(label="baseline", theory=theory, fit=fit_theory(theory, conditions, budget=BUDGET))]


def _generator(conditions, responses):
    client = FakeClient(responses)
    return XSLTheoryGenerator(llm_client=client, conditions=conditions, budget=BUDGET), client


def test_better_candidate_is_accepted_and_continue_stops(conditions, leaderboard, tmp_path) -> None:
    gen, client = _generator(conditions, [
        json.dumps(UNIFORM), critique("regenerate"),
        json.dumps(COUNTING), critique("continue"),
    ])
    result = gen.propose_theory(guide="g", leaderboard=leaderboard, max_iters=5, workspace=tmp_path)
    assert [it.accepted for it in result.history] == [True, True]
    assert result.best.candidate.name == "Counting"
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
    assert (tmp_path / "theories" / "baseline.yaml").exists()
    assert "baseline" in (tmp_path / "leaderboard.md").read_text()
