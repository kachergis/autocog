"""Phase 3: prospective designs that discriminate two fitted XSL theories.

No data are collected. A designer (an LLM) writes a *design generator*: Python
source for `make_design(knobs, rng)`, a few knobs with candidate values, and
a pre-stated *contrast*, a statistic of per-item test accuracy, e.g. the late
repetition gain at high minus low early exposure. A design has one or more
*arms*. Each arm is a between-subjects condition run on `PLANNED_N` people:
a training sequence plus an explicit m-AFC test with a word -> correct
object mapping, so asymmetric and non-1-to-1 designs are expressible.

A design is judged by how well its contrast would tell the two theories
apart in a real experiment. For each theory we simulate experiments with
`PLANNED_N` people per arm from its random-ability fit. Each simulated
experiment draws a parameter set from the full-corpus fit and the five
cross-validation fold fits (parameter uncertainty), then gives each person
an ability offset u ~ N(0, ability_sd^2) and samples their correct/incorrect
response to each item. The decision rule is pre-registered: the observed
contrast favours theory A if it lies on A's side of the midpoint between the
two theories' predicted (population-mean) contrasts. `power` is the smaller
of the two rates of choosing the true theory. A design is ACCEPTED at power
>= `POWER_TARGET`.

`search` tries the knob grid (several random orders per setting) with a
cheap simulation, then re-estimates the best setting with fresh, larger
simulations, so the reported power is not inflated by selecting on noise.
`freeze` records the accepted design with every leaderboard theory's
predictions, timestamped and hashed: the critical-experiment queue. Once a
queued design is run on people it becomes an ordinary corpus `Condition`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import signal
import subprocess
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import AliasChoices, Field, PrivateAttr

from src.promptable import Promptable
from src.xsl.corpus import Condition, MAFCTest
from src.xsl.fitting import ABILITY_SD, LAPSE, n_choices_per_item, population_accuracy
from src.xsl.scoring import predict_accuracy

# design space (proposal section 5): runnable online, comparable with the corpus
PLANNED_N = 50
POWER_TARGET = 0.8
MAX_WORDS = 36
MAX_OBJECTS = 36
MAX_TRIALS = 108
MAX_AFC = 36
MAX_TEST_ITEMS = 72
MAX_ARMS = 4
MIN_EXPOSURES = 2          # every word is heard on at least this many training trials
MAX_KNOB_VALUES = 12
MAX_COMBOS = 96            # knob settings tried per proposal (a random subset if the grid is larger)
SEEDS_PER_COMBO = 3        # random orders per knob setting
SEARCH_SIMS = 300          # simulated experiments per theory while searching
FINAL_SIMS = 2000          # ... for the final, independent power estimate
GENERATOR_SECONDS = 2.0

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]{0,47}$")


class DesignError(ValueError):
    """A design that cannot be run or violates the design space."""


@contextmanager
def _time_limit(seconds: float, what: str):
    """Abort LLM-written code that runs too long (Unix, main thread only)."""
    def handler(signum, frame):
        raise DesignError(f"{what} took longer than {seconds:g}s")
    try:
        old = signal.signal(signal.SIGALRM, handler)
    except ValueError:  # not the main thread: run unguarded
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# --- the designer's output ---------------------------------------------------------

class DesignProposal(Promptable):
    name: str = Field(..., description="Short slug for the design: lowercase letters, digits and underscores.")
    rationale: str = Field(
        ..., description=("Which mechanism difference between the two theories the design targets, and why "
                          "their predictions should diverge on the contrast."))
    generator_source: str = Field(
        ...,
        validation_alias=AliasChoices("generator", "generator_source"),
        description=("Python source defining def make_design(knobs, rng) -> dict. `knobs` maps each knob name "
                     "to one value; `rng` is a numpy Generator (use it for every random choice, so a seed "
                     "fixes the design). `np` is available. See DESIGN FORMAT for the dict to return."),
    )
    knobs: dict[str, list[Any]] = Field(
        ..., description=("Candidate values per knob, e.g. {\"early\": [0, 3, 6], \"load\": [2, 3]} "
                          "(at most 12 values each). Every combination is a candidate design; the search "
                          "keeps the most discriminating one. Use {} for a fixed design."))
    contrast_source: str = Field(
        ...,
        validation_alias=AliasChoices("contrast", "contrast_source"),
        description=("Python source defining def contrast(acc, groups) -> float. `acc` maps each arm name to "
                     "a numpy array of per-item accuracy (proportion correct, in test order); `groups` maps "
                     "each arm name to a numpy array of the items' group labels (strings). Example: "
                     "return acc['high'][groups['high'] == 'late'].mean() - acc['low'][groups['low'] == 'late'].mean()"),
    )
    prediction: str = Field(
        ..., description="What each theory predicts for the contrast (direction and rough size), and why.")

    _make: Any = PrivateAttr(default=None)
    _contrast: Any = PrivateAttr(default=None)

    def model_post_init(self, _context: Any) -> None:
        if not _SLUG.match(self.name):
            raise ValueError(f"name {self.name!r} must be a lowercase slug (letters, digits, underscores)")
        for knob, values in self.knobs.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"knob {knob!r} needs a non-empty list of candidate values")
            if len(values) > MAX_KNOB_VALUES:
                raise ValueError(f"knob {knob!r} has {len(values)} values (at most {MAX_KNOB_VALUES})")
        ns: dict[str, Any] = {"np": np}
        exec(self.generator_source, ns)
        if not callable(ns.get("make_design")):
            raise ValueError("generator_source must define a function named `make_design`")
        cs: dict[str, Any] = {"np": np}
        exec(self.contrast_source, cs)
        if not callable(cs.get("contrast")):
            raise ValueError("contrast_source must define a function named `contrast`")
        object.__setattr__(self, "_make", ns["make_design"])
        object.__setattr__(self, "_contrast", cs["contrast"])

    def settings(self, *, limit: int = MAX_COMBOS, seed: int = 0) -> list[dict[str, Any]]:
        """Knob settings to try: the full grid, or `limit` of them at random."""
        names = sorted(self.knobs)
        grid = [dict(zip(names, vals)) for vals in itertools.product(*(self.knobs[n] for n in names))]
        if len(grid) <= limit:
            return grid
        pick = np.random.default_rng(seed).choice(len(grid), size=limit, replace=False)
        return [grid[i] for i in sorted(pick)]

    def build(self, knobs: dict[str, Any], seed: int) -> "Design":
        rng = np.random.default_rng(seed)
        with _time_limit(GENERATOR_SECONDS, "make_design"):
            try:
                spec = self._make(dict(knobs), rng)
            except DesignError:
                raise
            except Exception as e:  # noqa: BLE001 - LLM-written code
                raise DesignError(f"make_design raised {type(e).__name__}: {e}") from e
        return design_from_spec(self.name, spec, knobs=knobs, seed=seed)

    def contrast_value(self, acc: dict[str, np.ndarray], groups: dict[str, np.ndarray]) -> float:
        try:
            v = float(self._contrast(acc, groups))
        except Exception as e:  # noqa: BLE001
            raise DesignError(f"contrast raised {type(e).__name__}: {e}") from e
        if not math.isfinite(v):
            raise DesignError(f"contrast returned {v}")
        return v


# --- designs -------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Arm:
    name: str
    condition: Condition
    groups: np.ndarray          # one label per test item

    def to_json(self) -> dict:
        c = self.condition
        return {
            "name": self.name,
            "n_words": len(c.words), "n_objects": len(c.objects), "n_trials": c.n_trials,
            "trials": [[list(w), list(o)] for w, o in zip(c.train_words, c.train_objects)],
            "test": [{"word": w, "candidates": list(os), "correct": k, "group": str(g)}
                     for w, os, k, g in zip(c.test.words, c.test.objects, c.test.correct, self.groups)],
        }


@dataclass(frozen=True, eq=False)
class Design:
    name: str
    knobs: dict[str, Any]
    seed: int
    arms: tuple[Arm, ...]

    def to_json(self) -> dict:
        return {"name": self.name, "knobs": self.knobs, "seed": self.seed,
                "arms": [a.to_json() for a in self.arms]}

    def summary(self) -> str:
        parts = []
        for a in self.arms:
            c = a.condition
            per = f"{np.mean([len(t) for t in c.train_words]):.1f}x{np.mean([len(t) for t in c.train_objects]):.1f}"
            afc = sorted({len(o) for o in c.test.objects})
            parts.append(f"{a.name}: {len(c.words)} words x {len(c.objects)} objects, {c.n_trials} trials "
                         f"({per} per trial), {len(c.test)} test items ({'/'.join(map(str, afc))}-AFC)")
        return "; ".join(parts)


def _labels(xs: Any, kind: str, where: str) -> tuple[str, ...]:
    if not isinstance(xs, (list, tuple)) or not xs:
        raise DesignError(f"{where}: {kind} must be a non-empty list")
    out = []
    for x in xs:
        if isinstance(x, (bool, float)) or not isinstance(x, (int, str, np.integer)):
            raise DesignError(f"{where}: {kind} labels must be ints or strings, got {x!r}")
        out.append(str(int(x)) if isinstance(x, (int, np.integer)) else x)
    if len(set(out)) != len(out):
        raise DesignError(f"{where}: repeated {kind} within one trial: {out}")
    return tuple(out)


def arm_from_spec(design: str, name: str, spec: dict) -> Arm:
    """Validate one arm against the design space and turn it into a Condition."""
    where = f"arm {name!r}"
    if not isinstance(spec, dict) or "trials" not in spec or "test" not in spec:
        raise DesignError(f"{where}: needs 'trials' and 'test'")
    trials = spec["trials"]
    if not isinstance(trials, (list, tuple)) or not 1 <= len(trials) <= MAX_TRIALS:
        raise DesignError(f"{where}: needs 1-{MAX_TRIALS} training trials, got "
                          f"{len(trials) if isinstance(trials, (list, tuple)) else trials!r}")
    words, objects = [], []
    for t, trial in enumerate(trials):
        if not isinstance(trial, (list, tuple)) or len(trial) != 2:
            raise DesignError(f"{where} trial {t}: must be [words, objects]")
        words.append(_labels(trial[0], "words", f"{where} trial {t}"))
        objects.append(_labels(trial[1], "objects", f"{where} trial {t}"))
    vocab = sorted({w for tw in words for w in tw})
    referents = sorted({o for to in objects for o in to})
    if len(vocab) > MAX_WORDS or len(referents) > MAX_OBJECTS:
        raise DesignError(f"{where}: {len(vocab)} words and {len(referents)} objects "
                          f"(at most {MAX_WORDS} and {MAX_OBJECTS})")
    exposures = {w: 0 for w in vocab}
    seen_with: dict[str, set[str]] = {w: set() for w in vocab}
    for tw, to in zip(words, objects):
        for w in tw:
            exposures[w] += 1
            seen_with[w].update(to)
    rare = [w for w, n in exposures.items() if n < MIN_EXPOSURES]
    if rare:
        raise DesignError(f"{where}: words heard on fewer than {MIN_EXPOSURES} trials: {rare[:8]}")

    test = spec["test"]
    if not isinstance(test, (list, tuple)) or not 1 <= len(test) <= MAX_TEST_ITEMS:
        raise DesignError(f"{where}: needs 1-{MAX_TEST_ITEMS} test items")
    t_words, t_objects, t_correct, groups = [], [], [], []
    for i, item in enumerate(test):
        iw = f"{where} test item {i}"
        if not isinstance(item, dict) or not {"word", "candidates", "correct"} <= set(item):
            raise DesignError(f"{iw}: needs 'word', 'candidates' and 'correct'")
        (w,) = _labels([item["word"]], "word", iw)
        cands = _labels(item["candidates"], "candidates", iw)
        (k,) = _labels([item["correct"]], "correct", iw)
        if w not in exposures:
            raise DesignError(f"{iw}: word {w} was never trained")
        if not 2 <= len(cands) <= MAX_AFC:
            raise DesignError(f"{iw}: {len(cands)} candidates (2-{MAX_AFC})")
        if any(c not in referents for c in cands):
            raise DesignError(f"{iw}: candidates include objects never shown in training")
        if k not in cands:
            raise DesignError(f"{iw}: correct object {k} is not among the candidates")
        if k not in seen_with[w]:
            raise DesignError(f"{iw}: correct object {k} never appeared with word {w} in training")
        t_words.append(w)
        t_objects.append(cands)
        t_correct.append(k)
        groups.append(str(item.get("group", "all")))
    cond = Condition(
        label=f"{design}:{name}", condition=f"proposed design {design}, arm {name}",
        train_words=tuple(words), train_objects=tuple(objects),
        test=MAFCTest(words=tuple(t_words), objects=tuple(t_objects), correct=tuple(t_correct)),
    )
    return Arm(name=name, condition=cond, groups=np.array(groups))


def design_from_spec(name: str, spec: Any, *, knobs: dict[str, Any] | None = None, seed: int = 0) -> Design:
    if not isinstance(spec, dict) or not isinstance(spec.get("arms"), dict) or not spec["arms"]:
        raise DesignError("make_design must return {'arms': {arm_name: {'trials': ..., 'test': ...}, ...}}")
    if len(spec["arms"]) > MAX_ARMS:
        raise DesignError(f"{len(spec['arms'])} arms (at most {MAX_ARMS})")
    arms = tuple(arm_from_spec(name, str(a), s) for a, s in spec["arms"].items())
    return Design(name=name, knobs=dict(knobs or {}), seed=seed, arms=arms)


# --- theories as data generators -----------------------------------------------------

@dataclass
class Generator:
    """A fitted theory as a source of simulated experiments: its learner, its
    full-corpus fit and its cross-validation fold fits."""

    label: str
    learner: Any
    params: dict[str, float]
    param_sets: list[dict[str, float]]

    @classmethod
    def from_entry(cls, entry) -> "Generator":
        return cls(label=entry.label, learner=entry.theory.learner(), params=dict(entry.fit.params),
                   param_sets=[dict(entry.fit.params)] + [dict(p) for p in entry.cv.fold_params])

    def item_accuracy(self, cond: Condition, params: dict[str, float]) -> np.ndarray:
        """Accuracy of an average-ability person per item, before guessing."""
        try:
            m = self.learner(params, cond.positional_trials, len(cond.words), len(cond.objects))
            p = predict_accuracy(np.asarray(m, dtype=float), cond, d=params.get("ch_dec"))
        except DesignError:
            raise
        except Exception as e:  # noqa: BLE001 - LLM-written learners may not handle every shape
            raise DesignError(f"{self.label} failed on {cond.label}: {type(e).__name__}: {e}") from e
        if p.shape != (cond.n_items,) or not np.all(np.isfinite(p)):
            raise DesignError(f"{self.label} gave invalid predictions on {cond.label}")
        return p

    def expected(self, design: Design) -> dict[str, np.ndarray]:
        """Population-mean per-item accuracy per arm (full-corpus fit)."""
        out = {}
        for a in design.arms:
            try:
                out[a.name] = population_accuracy(self.learner, self.params, a.condition)
            except Exception as e:  # noqa: BLE001
                raise DesignError(f"{self.label} failed on {a.condition.label}: {type(e).__name__}: {e}") from e
        return out


def simulate(proposal: DesignProposal, design: Design, gen: Generator, *, n_subj: int, n_sims: int,
             rng: np.random.Generator) -> np.ndarray:
    """Contrast values from `n_sims` simulated experiments with `n_subj` people per arm."""
    groups = {a.name: a.groups for a in design.arms}
    # logits, guessing rate, ability spread and choices per item, for each parameter set and arm
    latent = []
    for params in gen.param_sets:
        per_arm = {}
        for a in design.arms:
            p = np.clip(gen.item_accuracy(a.condition, params), 1e-6, 1 - 1e-6)
            per_arm[a.name] = (np.log(p) - np.log1p(-p), n_choices_per_item(a.condition))
        latent.append((per_arm, params.get(LAPSE, 0.0), params.get(ABILITY_SD, 0.0)))
    out = np.empty(n_sims)
    which = rng.integers(len(latent), size=n_sims)
    for s in range(n_sims):
        per_arm, lapse, sd = latent[which[s]]
        acc = {}
        for name, (eta, k) in per_arm.items():
            u = rng.normal(0.0, sd, size=(n_subj, 1))
            q = (1 - lapse) / (1 + np.exp(-(eta[None, :] + u))) + lapse / k[None, :]
            acc[name] = (rng.random(q.shape) < q).mean(axis=0)
        out[s] = proposal.contrast_value(acc, groups)
    return out


@dataclass
class Power:
    predicted: dict[str, float]          # population-mean contrast per theory
    threshold: float                     # midpoint: the pre-registered decision boundary
    hit_rate: dict[str, float]           # P(decision picks the theory | it generated the data)
    sim_sd: dict[str, float]             # spread of the simulated contrast
    n_subj: int
    n_sims: int

    @property
    def power(self) -> float:
        return min(self.hit_rate.values())

    def to_json(self) -> dict:
        return {"predicted": self.predicted, "threshold": self.threshold, "hit_rate": self.hit_rate,
                "sim_sd": self.sim_sd, "power": self.power, "n_subj": self.n_subj, "n_sims": self.n_sims}

    def summary(self) -> str:
        a, b = self.predicted
        return (f"predicted contrast {a} {self.predicted[a]:+.4f}, {b} {self.predicted[b]:+.4f} "
                f"(simulated SD {self.sim_sd[a]:.4f} / {self.sim_sd[b]:.4f} at N={self.n_subj} per arm); "
                f"decide {a} if the contrast is on its side of {self.threshold:+.4f}; "
                f"P(correct | {a}) {self.hit_rate[a]:.3f}, P(correct | {b}) {self.hit_rate[b]:.3f}; "
                f"power {self.power:.3f}")


def power_analysis(proposal: DesignProposal, design: Design, a: Generator, b: Generator, *,
                   n_subj: int = PLANNED_N, n_sims: int = SEARCH_SIMS, seed: int = 0) -> Power:
    groups = {x.name: x.groups for x in design.arms}
    predicted = {g.label: proposal.contrast_value(g.expected(design), groups) for g in (a, b)}
    threshold = 0.5 * (predicted[a.label] + predicted[b.label])
    sign = 1.0 if predicted[a.label] >= predicted[b.label] else -1.0
    rng = np.random.default_rng(seed)
    hit, sd = {}, {}
    for g, side in ((a, sign), (b, -sign)):
        sims = simulate(proposal, design, g, n_subj=n_subj, n_sims=n_sims, rng=rng)
        if predicted[a.label] == predicted[b.label]:
            hit[g.label] = 0.5
        else:
            hit[g.label] = float(np.mean((sims - threshold) * side > 0))
        sd[g.label] = float(np.std(sims))
    return Power(predicted=predicted, threshold=threshold, hit_rate=hit, sim_sd=sd, n_subj=n_subj, n_sims=n_sims)


# --- search -------------------------------------------------------------------------

@dataclass
class Trial:
    knobs: dict[str, Any]
    seed: int
    power: float | None
    error: str | None = None


@dataclass
class SearchResult:
    proposal: DesignProposal
    design: Design | None
    power: Power | None                  # final, independent estimate for `design`
    search_power: float | None           # the (optimistic) estimate the search selected on
    trials: list[Trial] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def accepted(self) -> bool:
        return self.power is not None and self.power.power >= POWER_TARGET

    @property
    def errors(self) -> list[str]:
        return sorted({t.error for t in self.trials if t.error})

    def summary(self) -> str:
        n_ok = sum(t.power is not None for t in self.trials)
        head = f"{self.proposal.name}: {n_ok}/{len(self.trials)} candidate designs valid"
        if self.design is None:
            return head + "; errors: " + "; ".join(self.errors[:5])
        knobs = ", ".join(f"{k}={v}" for k, v in self.design.knobs.items()) or "no knobs"
        out = (f"{head}; best setting {knobs} (seed {self.design.seed}): {self.design.summary()}.\n"
               f"  final estimate: {self.power.summary()}")
        if self.errors:
            out += "\n  some settings were invalid: " + "; ".join(self.errors[:3])
        return out


def search(proposal: DesignProposal, a: Generator, b: Generator, *, n_subj: int = PLANNED_N,
           max_combos: int = MAX_COMBOS, seeds: int = SEEDS_PER_COMBO, search_sims: int = SEARCH_SIMS,
           final_sims: int = FINAL_SIMS, seed: int = 0) -> SearchResult:
    """Try knob settings x random orders; re-estimate the best one independently."""
    t0 = time.perf_counter()
    trials: list[Trial] = []
    best: tuple[float, Design] | None = None
    for knobs in proposal.settings(limit=max_combos, seed=seed):
        for s in range(seeds):
            design_seed = seed * 1000 + s
            try:
                design = proposal.build(knobs, design_seed)
                pw = power_analysis(proposal, design, a, b, n_subj=n_subj, n_sims=search_sims,
                                    seed=seed * 7919 + s + 1)
            except DesignError as e:
                trials.append(Trial(knobs=knobs, seed=design_seed, power=None, error=str(e)))
                continue
            trials.append(Trial(knobs=knobs, seed=design_seed, power=pw.power))
            # ties go to the smaller experiment (fewer total trials)
            key = (pw.power, -sum(x.condition.n_trials for x in design.arms))
            if best is None or key > best[0]:
                best = (key, design)
    if best is None:
        return SearchResult(proposal=proposal, design=None, power=None, search_power=None, trials=trials,
                            seconds=time.perf_counter() - t0)
    design = best[1]
    final = power_analysis(proposal, design, a, b, n_subj=n_subj, n_sims=final_sims, seed=seed + 7919)
    return SearchResult(proposal=proposal, design=design, power=final, search_power=best[0][0],
                        trials=trials, seconds=time.perf_counter() - t0)


# --- the queue -----------------------------------------------------------------------

def _sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def freeze(result: SearchResult, entries: Sequence[Any], *, pair: tuple[str, str], leaderboard_path: Path,
           root: Path) -> dict:
    """The queue record for an accepted design: the design, the contrast and
    decision rule, and every leaderboard theory's frozen predictions."""
    assert result.design is not None and result.power is not None
    design, proposal = result.design, result.proposal
    groups = {a.name: a.groups for a in design.arms}
    predictions = {}
    for e in entries:
        g = Generator.from_entry(e)
        try:
            acc = g.expected(design)
            predictions[e.label] = {"items": {k: v.round(6).tolist() for k, v in acc.items()},
                                    "contrast": proposal.contrast_value(acc, groups)}
        except DesignError as err:
            predictions[e.label] = {"error": str(err)}
    body = {
        "design": design.to_json(),
        "proposal": proposal.model_dump(by_alias=False),
        "pair": list(pair),
        "decision_rule": {"contrast": proposal.contrast_source, "threshold": result.power.threshold,
                          "favours_above": max(result.power.predicted, key=result.power.predicted.get)},
        "power": result.power.to_json(),
        "planned_n_per_arm": result.power.n_subj,
        "predictions": predictions,
        "leaderboard": {"path": (str(leaderboard_path.relative_to(root))
                                 if leaderboard_path.is_relative_to(root) else str(leaderboard_path)),
                        "sha256": hashlib.sha256(leaderboard_path.read_bytes()).hexdigest()},
        "git_commit": _git_commit(root),
    }
    return {"id": f"{design.name}_{_sha256(body)[:10]}", "frozen_at": datetime.now(timezone.utc).isoformat(),
            "sha256": _sha256(body), **body}


# --- prompts -------------------------------------------------------------------------

DESIGN_SYSTEM = """\
You are an experimental psychologist designing a NEW cross-situational word \
learning experiment to tell two fitted theories apart. No data exist for it \
yet: the goal is a design on which the theories make clearly different, \
pre-registrable predictions that people can then be run on.

## THE PARADIGM
On each training trial a learner hears several novel words and sees several \
novel objects, with no indication of which word goes with which; each word \
has a consistent referent across trials. At test, people hear a word and \
choose its referent from candidate objects. Both theories are fitted to 47 \
existing adult conditions (between-subject ability differences modelled).

## WHAT TO PRODUCE
Python source for a design generator, `make_design(knobs, rng)`, a few knobs \
with candidate values, and a CONTRAST: a statistic of per-item test accuracy \
on which {a} and {b} disagree. The search builds the design at every knob \
setting (several random orders each), simulates experiments with {n} people \
per arm from each theory's fit (including parameter uncertainty), and keeps \
the setting where a pre-registered decision on the contrast (midpoint between \
the theories' predictions) most often picks the theory that generated the \
data. The design is ACCEPTED if that happens at least {power:.0%} of the time \
under both theories.

Aim for an effect with a clear mechanistic reading, not an artefact: a \
contrast between conditions or item types that differ in one factor. Arms \
are between-subject groups ({n} people each), so prefer within-arm item \
contrasts where possible (more power per person) and use at most {max_arms} \
arms.

## DESIGN FORMAT
`make_design` returns {{"arms": {{arm_name: arm, ...}}}}, where each arm is
  {{"trials": [[[word, ...], [object, ...]], ...],          # training, in order
    "test": [{{"word": w, "candidates": [o, ...], "correct": o, "group": "label"}}, ...]}}
Words and objects are integer (or string) labels, local to the arm. Each \
test item names its correct object explicitly and an optional group label \
the contrast can use.

## CONSTRAINTS (designs must be runnable online)
- per arm: at most {max_words} words, {max_objects} objects, {max_trials} training trials;
- no word or object repeated within a trial; any number of words and objects per trial \
(the corpus uses 1x2 up to 4x4; more words than objects or vice versa is allowed);
- every word is heard on at least {min_exp} training trials;
- test items: at most {max_items}; 2-{max_afc} candidates each, all shown during training; \
the correct object must have appeared with the word at least once. By default show all \
objects as candidates; use fewer only when the design needs it;
- `make_design` must finish in under {secs:g} s and use only `rng` for randomness.

## EXAMPLE (a two-arm frequency manipulation, not a recommendation)
```python
def make_design(knobs, rng):
    n = 18
    def arm(reps_hi):
        pairs = [(w, w) for w in range(n)]
        hi = set(range(n // 2))
        tokens = [p for p in pairs for _ in range(reps_hi if p[0] in hi else 3)]
        rng.shuffle(tokens)
        trials = [tokens[i:i + 3] for i in range(0, len(tokens) - 2, 3)]
        trials = [t for t in trials if len({{w for w, _ in t}}) == 3]
        test = [{{"word": w, "candidates": list(range(n)), "correct": w,
                 "group": "hi" if w in hi else "lo"}} for w in range(n)]
        return {{"trials": [[[w for w, _ in t], [o for _, o in t]] for t in trials], "test": test}}
    return {{"arms": {{"x": arm(knobs["reps"])}}}}
```
with knobs {{"reps": [6, 9]}} and contrast \
`def contrast(acc, groups): return acc['x'][groups['x'] == 'hi'].mean() - acc['x'][groups['x'] == 'lo'].mean()`.

## OUTPUT
{output_rules}
Return one JSON object with these fields:
{format}
"""


def _theory_block(n: int, entry) -> str:
    params = ", ".join(f"{k}={v:.4g}" for k, v in entry.fit.params.items())
    return (f"### THEORY {n}: {entry.label} — held-out NLL {entry.cv.heldout_nll:.1f}\n"
            f"Fitted parameters: {params}\n\n{entry.theory.pretty_print()}")


def render_design(*, a, b, corpus_table: str, debate_ledger: str, queue: Sequence[dict],
                  history: Sequence[str], n_subj: int, output_rules: str) -> tuple[str, str]:
    system = DESIGN_SYSTEM.format(
        a=a.label, b=b.label, n=n_subj, power=POWER_TARGET, max_arms=MAX_ARMS, max_words=MAX_WORDS,
        max_objects=MAX_OBJECTS, max_trials=MAX_TRIALS, min_exp=MIN_EXPOSURES, max_items=MAX_TEST_ITEMS,
        max_afc=MAX_AFC, secs=GENERATOR_SECONDS, output_rules=output_rules,
        format=DesignProposal.instruction_format())
    parts = [
        "## THEORIES\n" + _theory_block(1, a) + "\n\n" + _theory_block(2, b),
        "## WHERE THEY DIFFER ON THE EXISTING CORPUS (mean accuracy per condition)\n" + corpus_table,
        "## WHAT A CORPUS DEBATE BETWEEN THEM FOUND\n" + debate_ledger,
    ]
    if queue:
        parts.append("## DESIGNS ALREADY QUEUED (propose something different)\n" + "\n".join(
            f"- {q['design']['name']}: {q['proposal']['rationale']} (power {q['power']['power']:.2f})"
            for q in queue))
    if history:
        parts.append("## YOUR EARLIER ATTEMPTS FOR THIS DESIGN (not accepted; revise)\n" + "\n".join(
            f"- {h}" for h in history))
    return system, "\n\n".join(parts)


def chat_logged(llm_client, system: str, user: str, workspace: Path | None, log_label: str) -> str:
    result = llm_client.chat(messages=[{"role": "user", "content": user}], system=system, response_schema=None)
    if workspace is not None:
        prompts = workspace / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / f"{log_label}.md").write_text(
            f"# {log_label}\n\n## System Prompt\n\n{system}\n\n## User Prompt\n\n{user}\n\n"
            f"## Response\n\n{result.text}\n\n## Usage\n\n```json\n{json.dumps(result.usage, indent=2)}\n```\n")
    return result.text
