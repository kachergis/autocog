"""Loader tests for the XSLmodels corpus export in data/xsl/."""

from __future__ import annotations

import numpy as np
import pytest

from src.xsl.corpus import Condition, MAFCTest, load_corpus, load_holdout, load_manifest


@pytest.fixture(scope="module")
def corpus() -> list[Condition]:
    return load_corpus()


def test_corpus_size(corpus: list[Condition]) -> None:
    assert len(corpus) == 47
    assert len(corpus) == load_manifest()["n_conditions"]
    assert [c.index for c in corpus] == list(range(1, 48))


def test_every_condition_has_human_data(corpus: list[Condition]) -> None:
    for c in corpus:
        assert c.has_human_data
        assert c.n_subj and c.n_subj > 0
        assert len(c.accuracy) == c.n_items
        rm = c.response_matrix
        assert rm is not None
        assert rm.counts.shape == (c.n_items, len(c.objects))
        # each item's choice counts sum to at most the number of subjects
        assert (rm.counts.sum(axis=1) <= c.n_subj).all()


def test_trials_are_flat_label_tuples(corpus: list[Condition]) -> None:
    # condition 301 stores trials as one-row data.frames in R; the export
    # must still give flat tuples of labels
    c301 = next(c for c in corpus if c.label == "301")
    assert c301.train_words[0] == (11, 9)
    for c in corpus:
        for trial in c.train_words + c.train_objects:
            assert all(isinstance(x, int) for x in trial)


def test_asymmetric_condition_204(corpus: list[Condition]) -> None:
    c204 = next(c for c in corpus if c.label == "204")
    assert (len(c204.words), len(c204.objects)) == (18, 24)
    assert c204.test is None  # scored on the diagonal over all 24 objects


def test_positional_trials_index_sorted_vocab(corpus: list[Condition]) -> None:
    c = corpus[0]
    words, objects = c.positional_trials[0]
    assert [c.words[i] for i in words] == list(c.train_words[0])
    assert [c.objects[j] for j in objects] == list(c.train_objects[0])


def test_explicit_tests(corpus: list[Condition]) -> None:
    with_test = [c for c in corpus if c.test is not None]
    assert len(with_test) == 16
    assert sum(c.test.correct is not None for c in with_test) == 12
    for c in with_test:
        assert len(c.test.words) == len(c.test.objects) == c.n_items


def test_holdout_is_sealed() -> None:
    with pytest.raises(PermissionError):
        load_holdout()
    holdout = load_holdout(unseal=True)
    assert set(holdout) == {"benitez2020", "kachergis_initial_accuracy", "suanda2014", "koehne2013"}
    assert [len(v) for v in holdout.values()] == [7, 2, 3, 4]


def test_design_without_human_data() -> None:
    # a proposed design is a Condition with the human fields left empty
    d = Condition(label="new", condition="2x2", train_words=((1, 2), (1, 3)),
                  train_objects=((1, 2), (1, 3)))
    assert not d.has_human_data
    assert d.words == (1, 2, 3) and d.n_items == 3


def test_mismatched_trials_rejected() -> None:
    with pytest.raises(ValueError):
        Condition(label="bad", condition="", train_words=((1,),), train_objects=())


def test_accuracy_length_checked_against_test() -> None:
    test = MAFCTest(words=(1,), objects=((1, 2),))
    with pytest.raises(ValueError):
        Condition(label="bad", condition="", train_words=((1, 2),), train_objects=((1, 2),),
                  test=test, accuracy=np.array([0.5, 0.5]))
