"""The Python -> Rscript incumbent bridge.

The round-trip tests need R and the pinned XSLmodels library
(xsl-bakeoff/fits/v05/Rlib, or $XSL_RLIB) and are skipped without them.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from src.xsl import incumbents, models, scoring
from src.xsl.corpus import DATA_DIR, Condition, MAFCTest, load_corpus, load_manifest

ATOL = 1e-10

needs_r = pytest.mark.skipif(
    shutil.which("Rscript") is None or not (incumbents.default_rlib() / "XSLmodels").is_dir(),
    reason="needs Rscript and the pinned XSLmodels library",
)


def test_interim_panel_is_exported_and_deterministic() -> None:
    panel = incumbents.load_panel()
    assert [p.name for p in panel] == list(incumbents.INTERIM_PANEL)
    assert not any(p.stochastic for p in panel)
    # kachergis is nested in uncfam_gamma, which fits better
    every = incumbents.load_incumbents()
    assert every["uncfam_gamma"].sse_v05 < every["kachergis"].sse_v05


def _export_is_current() -> bool:
    return incumbents.pinned_commit() == load_manifest()["xslmodels_commit"]


@needs_r
def test_export_matches_pinned_library() -> None:
    """data/xsl/ must come from the XSLmodels build the bridge runs; when
    xsl-bakeoff re-pins, re-run scripts/xsl/export_corpus.R."""
    assert _export_is_current(), (
        f"data/xsl was exported from XSLmodels {load_manifest()['xslmodels_commit'][:7]} but the "
        f"pinned library is {(incumbents.pinned_commit() or 'unknown')[:7]}; re-run "
        "`Rscript scripts/xsl/export_corpus.R` once the v05 refit at that commit is complete"
    )


@needs_r
@pytest.mark.skipif(not _export_is_current(), reason="export is stale; see test_export_matches_pinned_library")
def test_bridge_reproduces_v05_predictions_on_corpus() -> None:
    corpus = load_corpus()
    every = incumbents.load_incumbents()
    chosen = incumbents.load_panel() + [every["kachergis"], every["baseline"]]
    preds = incumbents.predict(chosen, corpus)
    v05 = json.loads((DATA_DIR / "incumbents.json").read_text())
    for inc in chosen:
        for cond, got, want in zip(corpus, preds[inc.name], v05[inc.name]["v05_perf"], strict=True):
            np.testing.assert_allclose(got, np.array(want, dtype=float), rtol=0, atol=ATOL,
                                       err_msg=f"{inc.name}, condition {cond.label}")


@needs_r
def test_bridge_matches_python_port_on_a_new_asymmetric_design() -> None:
    # a prospective-style design outside the corpus: 36 words but only 12
    # objects (each object named by three synonyms), scored with an explicit
    # 12-AFC test over all objects
    rng = np.random.default_rng(0)
    words = np.arange(1, 37)
    referent = {w: 101 + (w - 1) % 12 for w in words}
    train_w, train_o = [], []
    for _ in range(4):
        order = rng.permutation(words)
        for k in range(0, 36, 3):
            tw = tuple(int(w) for w in order[k:k + 3])
            train_w.append(tw)
            train_o.append(tuple(sorted({referent[w] for w in tw})))
    all_objects = tuple(range(101, 113))
    design = Condition(
        label="synonyms", condition="3x<=3, 36w/12o",
        train_words=tuple(train_w), train_objects=tuple(train_o),
        test=MAFCTest(words=tuple(int(w) for w in words),
                      objects=tuple(all_objects for _ in words),
                      correct=tuple(referent[w] for w in words)),
    )
    kach = incumbents.load_incumbents()["kachergis"]
    r_perf = incumbents.predict([kach], [design])["kachergis"][0]
    py_perf, _ = scoring.run(models.uncfam, kach.params, design)
    assert r_perf.shape == (36,)
    np.testing.assert_allclose(py_perf, r_perf, rtol=0, atol=ATOL)
