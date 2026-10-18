"""Phase 2 debate: corpus queries, their scoring, and a full round with a fake LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest

import main_xsl_debate as debate_main
from src.xsl.corpus import load_corpus
from src.xsl.debate import SEPARATION, CorpusQuery, bootstrap_se, evaluate_query, human_accuracy
from src.xsl.fitting import FitBudget, cross_validate_theory, fit_theory
from src.xsl.generator import XSLTheoryGenerator
from src.xsl.prompts import Entry
from src.xsl.theory import XSLTheory

BUDGET = FitBudget(popsize=3, maxiter=3, polish=False, objective="subject")
COUNTING = {
    "name": "Counting",
    "theory": "Count co-occurrences.",
    "learn": ("def learn(params, trials, n_words, n_objects):\n"
              "    m = np.zeros((n_words, n_objects))\n"
              "    for w, o in trials:\n"
              "        m[np.ix_(w, o)] += 1\n"
              "    return m\n"),
    "parameters": {},
}


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def small(corpus):
    return [c for c in corpus if c.test is None][:4]


def test_query_compiles_and_evaluates(corpus) -> None:
    q = CorpusQuery(conditions=["221", "222"], metric="def metric(acc):\n    return acc['221'].mean() - acc['222'].mean()\n",
                    claim="x")
    acc = human_accuracy(corpus)
    assert q.value(acc) == pytest.approx(acc["221"].mean() - acc["222"].mean())
    with pytest.raises(ValueError):
        CorpusQuery(conditions=[], metric="def metric(acc):\n    return 0.0\n", claim="x")
    with pytest.raises(ValueError):
        CorpusQuery(conditions=["221"], metric="x = 1\n", claim="x")


def test_bootstrap_se_matches_the_analytic_se_of_a_condition_mean(corpus) -> None:
    c = next(x for x in corpus if x.label == "223")  # 77 people
    q = CorpusQuery(conditions=["223"], metric="def metric(acc):\n    return acc['223'].mean()\n", claim="x")
    per_person = np.nanmean(c.subject_correct, axis=1)
    analytic = per_person.std(ddof=0) / np.sqrt(len(per_person))
    se = bootstrap_se(q, {c.label: c}, n=2000, seed=1)
    assert se == pytest.approx(analytic, rel=0.08)
    assert bootstrap_se(q, {c.label: c}, n=200, seed=3) == bootstrap_se(q, {c.label: c}, n=200, seed=3)


def test_separability_and_winner(corpus) -> None:
    q = CorpusQuery(conditions=["221"], metric="def metric(acc):\n    return acc['221'].mean()\n", claim="x")
    human = human_accuracy(corpus)
    se = bootstrap_se(q, {c.label: c for c in corpus})
    far = {k: v + 10 * se for k, v in human.items()}
    near = {k: v + 0.1 * se for k, v in human.items()}
    out = evaluate_query(q, "A", corpus, {"A": human, "B": far})
    assert out.separable and out.winner == "A"
    assert out.errors_in_se()["A"] == pytest.approx(0.0, abs=1e-9)
    tie = evaluate_query(q, "A", corpus, {"A": human, "B": near})
    assert not tie.separable and tie.winner is None
    assert SEPARATION == 2.0


def test_unknown_conditions_are_rejected(corpus) -> None:
    q = CorpusQuery(conditions=["nope"], metric="def metric(acc):\n    return 0.0\n", claim="x")
    with pytest.raises(ValueError, match="unknown"):
        evaluate_query(q, "A", corpus, {"A": {}, "B": {}})


@dataclass
class _Result:
    text: str
    usage: dict = field(default_factory=dict)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages, system, response_schema=None):
        return _Result(text=self.responses.pop(0))


def _entry(label, theory, conds):
    return Entry(label=label, theory=theory, fit=fit_theory(theory, conds, budget=BUDGET),
                 cv=cross_validate_theory(theory, conds, k=2, budget=BUDGET))


def test_a_full_round_revises_the_losing_theory(small, tmp_path) -> None:
    labels = [c.label for c in small]
    uniform = XSLTheory(name="Uniform", theory="Nothing is learned.",
                        learn="def learn(params, trials, n_words, n_objects):\n    return np.ones((n_words, n_objects))\n")
    counting = XSLTheory(**COUNTING)
    a, b = _entry("xsl_01", counting, small), _entry("inc_uniform", uniform, small)
    entries = [a, b]
    by_label = {e.label: e for e in entries}
    # a statistic the two theories predict very differently: mean accuracy of the first condition
    query = json.dumps({"conditions": [labels[0]], "metric": f"def metric(acc):\n    return acc['{labels[0]}'].mean()\n",
                        "claim": "counting learns"})
    bad = json.dumps({"conditions": ["nope"], "metric": "def metric(acc):\n    return 0.0\n", "claim": "?"})
    verdict = json.dumps({"interpretation": "uniform learns nothing", "target": 2, "verdict": "new_theory",
                          "recommendation": "learn from co-occurrence"})
    candidate = json.dumps({**COUNTING, "name": "Counting v2"})
    critique = json.dumps({"interpretation": "i", "verdict": "continue", "rationale": "r"})
    client = FakeClient([query, bad, query, verdict, candidate, critique])
    gen = XSLTheoryGenerator(llm_client=client, conditions=small, budget=BUDGET, cv_folds=2)
    state = {"initial": ["xsl_01", "inc_uniform"], "slots": ["xsl_01", "inc_uniform"], "rounds": []}
    ledger = []
    record = debate_main.debate_round(gen, state, entries, by_label, small, ledger, tmp_path,
                                      query_attempts=2, max_iters=1)
    assert not client.responses
    assert [o["proposer"] for o in record["outcomes"]] == ["xsl_01", "inc_uniform"]
    assert record["failures"]["inc_uniform"][0].startswith("invalid query")
    for o in record["outcomes"]:  # the winner is whichever prediction is closer to people
        err = {k: abs(v - o["human"]) for k, v in o["predictions"].items()}
        assert set(err) == {"xsl_01", "inc_uniform"} and o["winner"] == min(err, key=err.get)
    assert record["target_label"] == "inc_uniform"
    # the revision beat the uniform learner, so it takes slot 2 under a fresh label
    assert record["revised_label"] == "xsl_02" and state["slots"] == ["xsl_01", "xsl_02"]
    assert [e.label for e in entries] == ["xsl_01", "inc_uniform", "xsl_02"]
    assert len(ledger) == 2
    debate_main.write_summary(tmp_path, state)
    assert "xsl_02 accepted" in (tmp_path / "debate.md").read_text()
    assert (tmp_path / "round_00" / "prompts" / "arbiter_attempt_00.md").exists()


def test_revision_must_beat_the_target(small, tmp_path) -> None:
    counting = XSLTheory(**COUNTING)
    a = _entry("xsl_01", counting, small)
    b = _entry("xsl_02", XSLTheory(**{**COUNTING, "name": "Counting B"}), small)
    entries, by_label = [a, b], {"xsl_01": a, "xsl_02": b}
    lab = small[0].label
    query = json.dumps({"conditions": [lab], "metric": f"def metric(acc):\n    return acc['{lab}'].mean()\n",
                        "claim": "c"})
    verdict = json.dumps({"interpretation": "i", "target": 1, "verdict": "new_model", "recommendation": "r"})
    worse = json.dumps({"name": "Uniform", "theory": "t",
                        "learn": "def learn(params, trials, n_words, n_objects):\n    return np.ones((n_words, n_objects))\n",
                        "parameters": {}})
    critique = json.dumps({"interpretation": "i", "verdict": "regenerate", "rationale": "r"})
    # identical predictions -> neither query is separable (2 attempts each)
    client = FakeClient([query, query, query, query, verdict, worse, critique])
    gen = XSLTheoryGenerator(llm_client=client, conditions=small, budget=BUDGET, cv_folds=2)
    state = {"initial": ["xsl_01", "xsl_02"], "slots": ["xsl_01", "xsl_02"], "rounds": []}
    record = debate_main.debate_round(gen, state, entries, by_label, small, [], tmp_path,
                                      query_attempts=2, max_iters=1)
    assert record["outcomes"] == []
    assert all(f.startswith("not separable") for f in record["failures"]["xsl_01"])
    assert record["revised_label"] is None and state["slots"] == ["xsl_01", "xsl_02"]
    assert [e.label for e in entries] == ["xsl_01", "xsl_02"]
