"""XSLTheory — a cross-situational word learning theory operationalised as code.

The XSL counterpart of `src.theory.Theory`. Training trials in XSL have no
response, so instead of a trial-by-trial `predict` / `policy` pair, a theory
supplies one `learn` function that maps a whole training sequence to a final
word x object association matrix:

    def learn(params, trials, n_words, n_objects) -> np.ndarray  # (n_words, n_objects)

`trials` is a list of `(words, objects)` pairs of 0-based integer index
arrays (the words heard and objects seen on each trial, in order). Test
choices are then made by the Luce rule shared by every XSLmodels model (see
`src.xsl.scoring`); a theory that declares a parameter named `ch_dec` gets the
power-Luce rule with that exponent.

Parameters are declared as intervals, "[lo, hi]", and are *fitted* to the
corpus (`src.xsl.fitting`), not sampled; a bare number pins a parameter.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
from pydantic import AliasChoices, Field, PrivateAttr

from src.promptable import Promptable
from src.xsl.corpus import Condition
from src.xsl.scoring import Learner, predict_accuracy, run

INDEX_SHUFFLE_SEED = 20261014

_INTERVAL = re.compile(r"^\s*\[\s*([^,\]]+)\s*,\s*([^,\]]+)\s*\]\s*$")


def parse_parameters(parameters: dict[str, Any]) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Split declared parameters into free bounds and pinned values.

    "[lo, hi]" (string or 2-element list) -> free in [lo, hi]; a bare number
    (or numeric string) -> pinned. Anything else is rejected.
    """
    free: dict[str, tuple[float, float]] = {}
    fixed: dict[str, float] = {}
    for name, spec in parameters.items():
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            lo, hi = (float(x) for x in spec)
        elif isinstance(spec, str) and (m := _INTERVAL.match(spec)):
            lo, hi = float(m.group(1)), float(m.group(2))
        else:
            try:
                fixed[name] = float(spec)
                continue
            except (TypeError, ValueError):
                raise ValueError(
                    f"parameter {name!r}: expected \"[lo, hi]\" or a number, got {spec!r}"
                ) from None
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"parameter {name!r}: need finite lo < hi, got [{lo}, {hi}]")
        free[name] = (lo, hi)
    if "lapse" in free or "lapse" in fixed:
        raise ValueError("'lapse' is reserved: a guessing rate is fitted for every theory automatically")
    return free, fixed


class XSLTheory(Promptable):
    """A theory of cross-situational word learning, with an executable learner."""

    name: str = Field(
        default="Unnamed XSL theory",
        description="Short name for the theory (a few words).",
    )
    description: str = Field(
        ...,
        validation_alias=AliasChoices("theory", "description"),
        description="The theoretical claim this theory makes about how people learn words across situations.",
    )
    learn_source: str = Field(
        ...,
        validation_alias=AliasChoices("learn", "learn_source"),
        description=(
            "Python source defining def learn(params, trials, n_words, n_objects) -> np.ndarray, "
            "returning the final (n_words, n_objects) association matrix. `np` is available."
        ),
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description='Free parameters as intervals, e.g. {"X": "[0.001, 5]"}; a bare number pins a value.',
    )
    rationale: str | None = Field(
        default=None,
        description="Why this theory should fit human cross-situational learning better than the alternatives.",
    )

    _learn_impl: Any = PrivateAttr(default=None)

    def model_post_init(self, _context: Any) -> None:
        ns: dict[str, Any] = {"np": np}
        exec(self.learn_source, ns)
        if not callable(ns.get("learn")):
            raise ValueError("learn_source must define a function named `learn`")
        object.__setattr__(self, "_learn_impl", ns["learn"])
        parse_parameters(self.parameters)  # validate eagerly

    @property
    def free_parameters(self) -> dict[str, tuple[float, float]]:
        return parse_parameters(self.parameters)[0]

    @property
    def fixed_parameters(self) -> dict[str, float]:
        return parse_parameters(self.parameters)[1]

    @property
    def n_free(self) -> int:
        return len(self.free_parameters)

    def learner(self) -> Learner:
        """The theory's `learn`, shown each condition under shuffled indices.

        In the corpus, word i's referent is object i (sorted order), so a
        learner could score well just by favouring the matrix diagonal. The
        wrapper relabels words and objects with a fixed pseudo-random
        permutation before calling `learn` and maps the returned matrix back.
        A theory that only uses co-occurrence is unaffected (up to
        floating-point summation order); one that leans on index identity
        loses that information.
        """
        impl = self._learn_impl

        def learn(params: dict[str, Any], trials, n_words: int, n_objects: int) -> np.ndarray:
            rng = np.random.default_rng([INDEX_SHUFFLE_SEED, n_words, n_objects, len(trials)])
            new_w, new_o = rng.permutation(n_words), rng.permutation(n_objects)
            shown = [(new_w[w], new_o[o]) for w, o in trials]
            m = np.asarray(impl(params, shown, n_words, n_objects), dtype=float)
            if m.shape != (n_words, n_objects):
                return m  # let the caller report the shape error
            return m[np.ix_(new_w, new_o)]

        return learn

    def full_params(self, free_values: dict[str, float]) -> dict[str, float]:
        return {**self.fixed_parameters, **free_values}

    def midpoint(self) -> dict[str, float]:
        return self.full_params({k: (lo + hi) / 2 for k, (lo, hi) in self.free_parameters.items()})

    def check(self, conditions: list[Condition], *, max_seconds: float = 2.0) -> float:
        """Run the learner once on `conditions` at the parameter midpoint.

        Raises if it errors, returns the wrong shape, non-finite or negative
        associations, or is not deterministic (two runs disagree). Returns
        the seconds taken, and raises if that exceeds `max_seconds`: the
        fitter calls the learner thousands of times.
        """
        learn = self.learner()
        params = self.midpoint()
        t0 = time.perf_counter()
        for c in conditions:
            m = learn(params, c.positional_trials, len(c.words), len(c.objects))
            if m.shape != (len(c.words), len(c.objects)):
                raise ValueError(
                    f"learn returned shape {m.shape} for condition {c.label}; "
                    f"expected ({len(c.words)}, {len(c.objects)})"
                )
            if not np.all(np.isfinite(m)) or (m < 0).any():
                raise ValueError(f"learn returned non-finite or negative associations on {c.label}")
            predict_accuracy(m, c, d=params.get("ch_dec"))
        elapsed = time.perf_counter() - t0
        _, m2 = run(learn, params, conditions[0])
        _, m1 = run(learn, params, conditions[0])
        if not np.array_equal(m1, m2):
            raise ValueError("learn is not deterministic: two runs with the same parameters differ")
        if elapsed > max_seconds:
            raise ValueError(
                f"learn took {elapsed:.2f}s for one pass over {len(conditions)} conditions "
                f"(limit {max_seconds}s); it must be fast enough to fit"
            )
        return elapsed

    def to_yaml(self, path) -> None:
        """Write in seed-YAML format (`XSLTheory.from_yaml` reads it back)."""
        import yaml

        class _Dumper(yaml.SafeDumper):
            pass

        def _str(dumper, s):  # multi-line strings (the learn source) as literal blocks
            style = "|" if "\n" in s else None
            return dumper.represent_scalar("tag:yaml.org,2002:str", s, style=style)

        _Dumper.add_representer(str, _str)
        out = {"name": self.name, "theory": self.description, "learn": self.learn_source,
               "parameters": self.parameters, "rationale": self.rationale}
        with open(path, "w") as f:
            yaml.dump(out, f, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)

    def pretty_print(self) -> str:
        params = "\n".join(f"- {k}: {v}" for k, v in self.parameters.items()) or "(none)"
        return (
            f"**Name:** {self.name}\n\n"
            f"**Description:** {self.description}\n\n"
            f"**Parameters:**\n{params}\n\n"
            f"**`learn` source:**\n```python\n{self.learn_source.rstrip()}\n```\n"
        )
