"""The XSLmodels human corpus, loaded from the JSON export in `data/xsl/`.

`scripts/xsl/export_corpus.R` writes the export from the XSLmodels build pinned
for the xsl-bakeoff v05 fits; `manifest.json` records the commit. Nothing here
needs R.

A `Condition` is one cross-situational word learning experiment: a training
sequence (the words and objects shown on each trial), an optional explicit
test, and — for collected data — per-item human accuracy and choice counts. A
proposed design that has not been run yet is the same object with the human
fields left empty, so a prospective experiment becomes a corpus condition
unchanged once its data comes in.

Item scoring follows XSLmodels:

  * no `test`: one item per word, scored as an n-AFC over every object, with
    word i's correct referent at sorted position i (the corpus's diagonal
    convention);
  * with a `test`: one item per test trial — a probe word, its candidate
    objects, and optionally an explicit correct object (otherwise the object
    with the probe word's label).

The scorable holdout sets are sealed: `load_holdout` refuses to load them
unless the caller passes `unseal=True`, so they cannot leak into a loop by
accident. They are for final evaluation only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "xsl"

Label = int | str
Trial = tuple[Label, ...]


@dataclass(frozen=True)
class MAFCTest:
    """Explicit m-AFC test: one probe word and its candidate objects per trial."""

    words: tuple[Label, ...]
    objects: tuple[Trial, ...]
    correct: tuple[Label, ...] | None = None

    def __len__(self) -> int:
        return len(self.words)


# eq=False: these hold numpy arrays, which have no single truth value.
@dataclass(frozen=True, eq=False)
class ResponseMatrix:
    """Human choice counts: one row per test item, one column per object."""

    items: tuple[str, ...]
    objects: tuple[str, ...]
    counts: np.ndarray


@dataclass(frozen=True, eq=False)
class Condition:
    label: str
    condition: str
    train_words: tuple[Trial, ...]
    train_objects: tuple[Trial, ...]
    test: MAFCTest | None = None
    accuracy: np.ndarray | None = None
    n_subj: int | None = None
    response_matrix: ResponseMatrix | None = None
    description: str = ""
    index: int | None = None
    # per-subject test responses, (n_subj, n_items) of 1 = correct, 0 = wrong,
    # NaN = not tested; from data/xsl/subject_responses.json when available
    subject_correct: np.ndarray | None = None

    def __post_init__(self) -> None:
        if len(self.train_words) != len(self.train_objects):
            raise ValueError(
                f"{self.label}: {len(self.train_words)} word trials but "
                f"{len(self.train_objects)} object trials"
            )
        if self.accuracy is not None and len(self.accuracy) != self.n_items:
            raise ValueError(
                f"{self.label}: {len(self.accuracy)} accuracies for {self.n_items} items"
            )

    @cached_property
    def words(self) -> tuple[Label, ...]:
        """Sorted vocabulary; a word's matrix row is its position here."""
        return tuple(sorted({w for trial in self.train_words for w in trial}))

    @cached_property
    def objects(self) -> tuple[Label, ...]:
        """Sorted referents; an object's matrix column is its position here."""
        return tuple(sorted({o for trial in self.train_objects for o in trial}))

    @cached_property
    def positional_trials(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Each trial as (word rows, object columns), 0-based.

        Mirrors XSLmodels' `positional_train`: models index the association
        matrix by position in the sorted vocabularies, never by raw label.
        """
        w_pos = {w: i for i, w in enumerate(self.words)}
        o_pos = {o: i for i, o in enumerate(self.objects)}
        return tuple(
            (
                np.array([w_pos[w] for w in tw], dtype=np.intp),
                np.array([o_pos[o] for o in to], dtype=np.intp),
            )
            for tw, to in zip(self.train_words, self.train_objects)
        )

    @property
    def n_trials(self) -> int:
        return len(self.train_words)

    @property
    def n_items(self) -> int:
        return len(self.test) if self.test is not None else len(self.words)

    @property
    def has_human_data(self) -> bool:
        return self.accuracy is not None


def _condition_from_json(d: dict[str, Any]) -> Condition:
    test = None
    if d.get("test"):
        t = d["test"]
        correct = tuple(c for (c,) in t["correct"]) if "correct" in t else None
        test = MAFCTest(
            words=tuple(w for (w,) in t["words"]),
            objects=tuple(tuple(os) for os in t["objects"]),
            correct=correct,
        )
    rm = d.get("response_matrix")
    return Condition(
        index=d.get("index"),
        label=str(d["label"]),
        condition=d.get("condition", ""),
        description=d.get("description") or "",
        train_words=tuple(tuple(tr) for tr in d["train"]["words"]),
        train_objects=tuple(tuple(tr) for tr in d["train"]["objects"]),
        test=test,
        accuracy=None if d.get("accuracy") is None else np.asarray(d["accuracy"], dtype=float),
        n_subj=None if d.get("n_subj") is None else int(d["n_subj"]),
        response_matrix=None if rm is None else ResponseMatrix(
            items=tuple(rm["items"]),
            objects=tuple(rm["objects"]),
            counts=np.asarray(rm["counts"], dtype=float),
        ),
    )


def _read(name: str, data_dir: Path | None) -> Any:
    path = (data_dir or DATA_DIR) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `Rscript scripts/xsl/export_corpus.R` from the repo root"
        )
    return json.loads(path.read_text())


def load_corpus(data_dir: Path | None = None, *, subjects: bool = True) -> list[Condition]:
    """The fitting corpus (XSLmodels `xsl_datasets`), in package order.

    With `subjects` (the default) each condition also gets its per-subject
    responses (`Condition.subject_correct`) if `subject_responses.json` has
    been exported (`scripts/xsl/export_subject_responses.R`). They are checked
    against the condition: one row per subject, one column per item, and the
    column means reproduce the stored per-item accuracy.
    """
    conditions = [_condition_from_json(d) for d in _read("corpus.json", data_dir)]
    path = (data_dir or DATA_DIR) / "subject_responses.json"
    if not subjects or not path.exists():
        return conditions
    by_label = {d["label"][0] if isinstance(d["label"], list) else d["label"]: d
                for d in json.loads(path.read_text())}
    out = []
    for c in conditions:
        d = by_label.get(c.label)
        if d is None:
            out.append(c)
            continue
        m = np.array([[np.nan if v is None else float(v) for v in row] for row in d["correct"]])
        if m.shape != (c.n_subj, c.n_items):
            raise ValueError(f"{c.label}: subject responses {m.shape}, expected ({c.n_subj}, {c.n_items})")
        if not np.allclose(np.nanmean(m, axis=0), c.accuracy, atol=1e-9):
            raise ValueError(f"{c.label}: subject responses do not reproduce the per-item accuracy")
        out.append(replace(c, subject_correct=m))
    return out


def load_holdout(*, unseal: bool = False, data_dir: Path | None = None) -> dict[str, list[Condition]]:
    """The scorable holdout sets, keyed by name. Final evaluation only."""
    if not unseal:
        raise PermissionError(
            "The XSL holdout sets are sealed for final evaluation; "
            "pass unseal=True only from an evaluation script."
        )
    return {
        s["name"]: [_condition_from_json(d) for d in s["conditions"]]
        for s in _read("holdout.json", data_dir)
    }


def load_manifest(data_dir: Path | None = None) -> dict[str, Any]:
    return _read("manifest.json", data_dir)
