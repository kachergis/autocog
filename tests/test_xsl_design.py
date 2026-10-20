"""Phase 3 prospective designs: validation, power simulation, search, the queue."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

import main_xsl_design as design_main
from src.xsl.design import (
    MAX_ARMS, MIN_EXPOSURES, DesignError, DesignProposal, Generator, design_from_spec, freeze,
    power_analysis, search, simulate,
)
from src.xsl.theory import XSLTheory

N = 6
TEST = [{"word": w, "candidates": list(range(N)), "correct": w, "group": "a" if w < 3 else "b"} for w in range(N)]

# one arm per value of `reps`: consistent 2x2 trials, each word heard `reps` times
GEN = """
def make_design(knobs, rng):
    n = 6
    trials = []
    for _ in range(knobs["reps"]):
        ws = [int(x) for x in rng.permutation(n)]
        trials += [[ws[i:i + 2], ws[i:i + 2]] for i in range(0, n, 2)]
    test = [{"word": w, "candidates": list(range(n)), "correct": w} for w in range(n)]
    return {"arms": {"x": {"trials": trials, "test": test}}}
"""
MEAN = "def contrast(acc, groups):\n    return float(acc['x'].mean())\n"


def proposal(**kw) -> DesignProposal:
    base = dict(name="reps", rationale="r", generator=GEN, knobs={"reps": [2, 4]}, contrast=MEAN, prediction="p")
    return DesignProposal(**{**base, **kw})


def counting(params, trials, n_words, n_objects):
    m = np.zeros((n_words, n_objects))
    for w, o in trials:
        m[np.ix_(w, o)] += 1
    return m


def uniform(params, trials, n_words, n_objects):
    return np.ones((n_words, n_objects))


def gen(label, learner, sd=1.0, lapse=0.001, sets=1) -> Generator:
    params = {"lapse": lapse, "ability_sd": sd}
    return Generator(label=label, learner=learner, params=params, param_sets=[params] * sets)


def two_by_two(reps=2):
    trials = [[[0, 1], [0, 1]], [[2, 3], [2, 3]], [[4, 5], [4, 5]]] * reps
    return {"trials": trials, "test": TEST}


# --- validation ----------------------------------------------------------------------

def test_a_valid_design_becomes_corpus_conditions() -> None:
    d = design_from_spec("demo", {"arms": {"x": two_by_two()}})
    (arm,) = d.arms
    c = arm.condition
    assert c.label == "demo:x" and c.n_trials == 6 and c.n_items == N and not c.has_human_data
    assert list(arm.groups) == ["a"] * 3 + ["b"] * 3
    assert c.test.correct == tuple(str(w) for w in range(N))
    assert design_from_spec("demo", json.loads(json.dumps({"arms": {"x": two_by_two()}}))).to_json() == d.to_json()


@pytest.mark.parametrize("spec, msg", [
    ({"arms": {}}, "make_design must return"),
    ({"arms": {str(i): two_by_two() for i in range(MAX_ARMS + 1)}}, "arms"),
    ({"arms": {"x": {"trials": [[[0, 0], [0, 1]]] * 2, "test": TEST}}}, "repeated"),
    ({"arms": {"x": {"trials": [[[i], [i]] for i in range(40)] * 2, "test": TEST}}}, "at most"),
    ({"arms": {"x": two_by_two(reps=MIN_EXPOSURES - 1)}}, "fewer than"),
    ({"arms": {"x": {**two_by_two(), "test": [{"word": 0, "candidates": [0, 9], "correct": 0}]}}}, "never shown"),
    ({"arms": {"x": {**two_by_two(), "test": [{"word": 0, "candidates": [0, 2], "correct": 2}]}}}, "never appeared"),
    ({"arms": {"x": {**two_by_two(), "test": [{"word": 0, "candidates": [1, 2], "correct": 0}]}}}, "not among"),
    ({"arms": {"x": {**two_by_two(), "test": [{"word": 9, "candidates": [0, 1], "correct": 0}]}}}, "never trained"),
    ({"arms": {"x": {"trials": [[[0.5], [0]]] * 2, "test": TEST}}}, "ints or strings"),
])
def test_invalid_designs_are_rejected(spec, msg) -> None:
    with pytest.raises(DesignError, match=msg):
        design_from_spec("bad", spec)


def test_proposals_compile_and_enumerate_settings() -> None:
    p = proposal(knobs={"a": [1, 2, 3], "b": ["x", "y"]})
    assert len(p.settings()) == 6 and {"a": 1, "b": "x"} in p.settings()
    assert len(p.settings(limit=4)) == 4 and p.settings(limit=4) == p.settings(limit=4)
    with pytest.raises(ValueError, match="slug"):
        proposal(name="Bad Name")
    with pytest.raises(ValueError, match="make_design"):
        proposal(generator="x = 1\n")
    with pytest.raises(ValueError, match="at most"):
        proposal(knobs={"a": list(range(20))})
    with pytest.raises(DesignError, match="make_design raised"):
        proposal(knobs={"other": [1]}).build({"other": 1}, 0)


def test_runaway_generators_are_stopped() -> None:
    p = proposal(generator="def make_design(knobs, rng):\n    while True:\n        pass\n")
    with pytest.raises(DesignError, match="longer than"):
        p.build({"reps": 2}, 0)


# --- simulation and power ------------------------------------------------------------

def test_simulated_experiments_average_to_the_population_prediction() -> None:
    p = proposal()
    d = p.build({"reps": 2}, 0)
    g = gen("count", counting, sd=1.5)
    expected = p.contrast_value(g.expected(d), {"x": d.arms[0].groups})
    sims = simulate(p, d, g, n_subj=50, n_sims=2000, rng=np.random.default_rng(0))
    assert abs(sims.mean() - expected) < 4 * sims.std() / np.sqrt(len(sims))
    # sd shrinks roughly with sqrt(N)
    big = simulate(p, d, g, n_subj=200, n_sims=2000, rng=np.random.default_rng(1))
    assert big.std() == pytest.approx(sims.std() / 2, rel=0.2)


def test_power_separates_different_theories_and_not_identical_ones() -> None:
    p = proposal()
    d = p.build({"reps": 3}, 0)
    a, b = gen("count", counting), gen("flat", uniform)
    pw = power_analysis(p, d, a, b, n_subj=50, n_sims=500)
    assert pw.predicted["count"] > pw.predicted["flat"]
    assert pw.power > 0.95 and pw.threshold == pytest.approx(np.mean(list(pw.predicted.values())))
    same = power_analysis(p, d, a, gen("count2", counting), n_subj=50, n_sims=200)
    assert same.power == 0.5
    assert power_analysis(p, d, a, b, n_sims=200, seed=4).hit_rate == power_analysis(p, d, a, b, n_sims=200, seed=4).hit_rate


def test_failing_learners_make_a_design_invalid_not_a_crash() -> None:
    def broken(params, trials, n_words, n_objects):
        raise IndexError("square matrices only")
    p = proposal(knobs={"reps": [2]})
    r = search(p, gen("count", counting), gen("broken", broken), seeds=1, search_sims=50, final_sims=50)
    assert r.design is None and not r.accepted and "broken failed" in r.errors[0]


def practice(params, trials, n_words, n_objects):
    """Accuracy grows with the number of training trials: (T + 1) / (T + n_objects)."""
    return np.eye(n_words, n_objects) * len(trials) + 1.0


def test_search_keeps_the_most_discriminating_setting_and_reestimates_it() -> None:
    # more repetitions -> `practice` pulls away from chance; heavy guessing keeps power off its ceiling
    p = proposal(knobs={"reps": [2, 6]})
    r = search(p, gen("practice", practice, lapse=0.9), gen("flat", uniform, lapse=0.9), seeds=2,
               search_sims=300, final_sims=400)
    assert r.design.knobs == {"reps": 6} and r.power.n_sims == 400 and len(r.trials) == 4
    assert r.accepted == (r.power.power >= 0.8)


def test_ties_go_to_the_smaller_experiment() -> None:
    # counting on re-paired 2x2 trials stays at .5 whatever the repetitions: both settings tie at power 1
    r = search(proposal(knobs={"reps": [6, 2]}), gen("count", counting), gen("flat", uniform), seeds=1,
               search_sims=100, final_sims=100)
    assert [t.power for t in r.trials] == [1.0, 1.0] and r.design.knobs == {"reps": 2}


def test_freeze_records_every_theory_and_a_stable_hash(tmp_path) -> None:
    p = proposal(knobs={"reps": [4]})
    a, b = gen("count", counting), gen("flat", uniform)
    r = search(p, a, b, seeds=1, search_sims=100, final_sims=200)
    lb = tmp_path / "leaderboard.json"
    lb.write_text("[]")

    def entry(label, learner):
        return SimpleNamespace(label=label, theory=SimpleNamespace(learner=lambda: learner),
                               fit=SimpleNamespace(params={"lapse": 0.001, "ability_sd": 1.0}),
                               cv=SimpleNamespace(fold_params=[]))

    def broken(params, trials, n_words, n_objects):
        raise ValueError("nope")

    rec = freeze(r, [entry("count", counting), entry("flat", uniform), entry("broken", broken)],
                 pair=("count", "flat"), leaderboard_path=lb, root=tmp_path)
    assert rec["id"].startswith("reps_") and len(rec["sha256"]) == 64
    assert set(rec["predictions"]) == {"count", "flat", "broken"} and "error" in rec["predictions"]["broken"]
    assert rec["decision_rule"]["favours_above"] == "count"
    assert rec["predictions"]["count"]["contrast"] == pytest.approx(rec["power"]["predicted"]["count"])
    again = freeze(r, [entry("count", counting), entry("flat", uniform), entry("broken", broken)],
                   pair=("count", "flat"), leaderboard_path=lb, root=tmp_path)
    assert again["sha256"] == rec["sha256"]
    json.dumps(rec)


# --- the designer loop ---------------------------------------------------------------

@dataclass
class _Result:
    text: str
    usage: dict = field(default_factory=dict)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages, system, response_schema=None):
        return _Result(text=self.responses.pop(0))


def test_design_slot_revises_until_accepted(tmp_path) -> None:
    def entry(label, learner_src, learner):
        theory = XSLTheory(name=label, theory="t", learn=learner_src)
        return SimpleNamespace(label=label, theory=theory,
                               fit=SimpleNamespace(params={"lapse": 0.001, "ability_sd": 1.0}),
                               cv=SimpleNamespace(fold_params=[], heldout_nll=1.0))
    count_src = ("def learn(params, trials, n_words, n_objects):\n    m = np.zeros((n_words, n_objects))\n"
                 "    for w, o in trials:\n        m[np.ix_(w, o)] += 1\n    return m\n")
    flat_src = "def learn(params, trials, n_words, n_objects):\n    return np.ones((n_words, n_objects))\n"
    a, b = entry("count", count_src, counting), entry("flat", flat_src, uniform)
    gens = (Generator.from_entry(a), Generator.from_entry(b))
    # identical arms -> no contrast; then a contrast that the two learners disagree on
    flat_contrast = "def contrast(acc, groups):\n    return 0.0\n"
    weak = json.dumps({"name": "weak", "rationale": "r", "generator": GEN, "knobs": {"reps": [2]},
                       "contrast": flat_contrast, "prediction": "p"})
    strong = json.dumps({"name": "strong", "rationale": "r", "generator": GEN, "knobs": {"reps": [2, 4]},
                         "contrast": MEAN, "prediction": "p"})
    client = FakeClient(["not json", weak, strong])
    lb = tmp_path / "lb"
    lb.mkdir()
    (lb / "leaderboard.json").write_text("[]")
    args = SimpleNamespace(max_iters=3, proposal_attempts=2, n_subj=50, max_combos=10, seeds=1,
                           search_sims=100, final_sims=300, leaderboard=lb)
    attempts: list = []
    rec = design_main.design_slot(client, 0, a, b, gens, [a, b], corpus_table="(table)", ledger="(ledger)",
                                  queue=[], attempts=attempts, out=tmp_path, args=args)
    assert not client.responses
    assert [x["accepted"] for x in attempts] == [False, True]
    assert rec is not None and rec["design"]["name"] == "strong" and rec["power"]["power"] >= 0.8
    second_prompt = (tmp_path / "design_00" / "prompts" / "design_iter_01_attempt_00.md").read_text()
    assert "EARLIER ATTEMPTS" in second_prompt and "power 0.500" in second_prompt
    design_main.write_queue_md(tmp_path, [rec], attempts)
    assert "Decision rule" in (tmp_path / "queue.md").read_text()
