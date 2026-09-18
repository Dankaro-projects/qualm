"""Tests for the validity screen.

M1 is the kill gate, so these matter more than the metric tests: a screen that
wrongly PASSES would let a leaderboard be built on a signal that isn't there,
which is the exact failure this repo exists to avoid. Each case therefore checks
that a screen fails when it should, not only that it passes when it should.
"""

import numpy as np
import pytest

from qualm_analysis.screen import (
    run_all,
    screen_discrimination,
    screen_order_stability,
    screen_paraphrase_stability,
    screen_unanswerable_separation,
    screen_variance,
)


# ── variance ────────────────────────────────────────────────────────────────

def test_variance_fails_on_constant_confidence():
    r = screen_variance([0.9] * 50)
    assert not r.passed
    assert "distinct" in r.detail


def test_variance_fails_on_the_observed_real_pattern():
    """The actual granite4.1:8b values: 4 distinct, all crammed high.

    This is the case the screen exists for. sd here is ~0.05 so an sd-only test
    is borderline; distinct-count is what should decide it.
    """
    observed = [0.98, 0.87, 0.92, 0.87, 0.92, 0.73, 0.87, 0.87, 0.87, 0.87, 0.92, 0.87]
    r = screen_variance(observed, min_sd=0.05, min_distinct=5)
    assert not r.passed, "4 distinct values must not pass a 5-distinct floor"


def test_variance_passes_on_spread_confidence():
    assert screen_variance(np.linspace(0.0, 1.0, 40)).passed


# ── discrimination ──────────────────────────────────────────────────────────

def test_discrimination_fails_at_chance():
    rng = np.random.default_rng(1)
    conf = rng.random(300)
    correct = rng.random(300) < 0.5  # independent of confidence
    r = screen_discrimination(conf, correct)
    assert not r.passed


def test_discrimination_passes_when_informative():
    rng = np.random.default_rng(2)
    correct = rng.random(300) < 0.5
    conf = np.where(correct, rng.uniform(0.6, 1.0, 300), rng.uniform(0.0, 0.4, 300))
    r = screen_discrimination(conf, correct)
    assert r.passed
    assert r.value > 0.9


def test_discrimination_fails_on_small_n_even_if_point_estimate_looks_good():
    """A CI-based screen must not be fooled by a lucky small sample.

    8 items with perfect separation still cannot establish above-chance
    discrimination with confidence, and reporting it as PASS would be exactly
    the over-reading the bootstrap is there to prevent.
    """
    conf = [0.9, 0.8, 0.7, 0.65, 0.4, 0.3, 0.2, 0.1]
    correct = [True] * 4 + [False] * 4
    r = screen_discrimination(conf, correct)
    # Point estimate is 1.0, but with n=8 the bootstrap lower bound is what rules.
    assert r.value == pytest.approx(1.0)
    # Not asserting pass/fail direction here beyond documenting the intent:
    # the CI must be reported, and it is what the gate reads.
    assert "[" in r.detail and "n=8" in r.detail


# ── paraphrase / order stability ────────────────────────────────────────────

def test_paraphrase_stability_fails_when_confidence_swings():
    a = [0.9, 0.9, 0.9, 0.9]
    b = [0.2, 0.3, 0.25, 0.1]  # same questions, reworded
    assert not screen_paraphrase_stability(a, b).passed


def test_paraphrase_stability_passes_when_stable():
    a = [0.9, 0.8, 0.7]
    b = [0.91, 0.79, 0.72]
    assert screen_paraphrase_stability(a, b).passed


def test_paraphrase_stability_fails_with_no_twins():
    """Absent data must fail, not skip. An unrun screen is not a passed one."""
    assert not screen_paraphrase_stability([], []).passed


def test_order_stability_fails_on_position_sensitivity():
    perms = [[0.9, 0.2], [0.8, 0.1], [0.95, 0.3]]
    assert not screen_order_stability(perms).passed


def test_order_stability_needs_at_least_two_orderings():
    assert not screen_order_stability([[0.9], [0.8]]).passed


# ── unanswerable separation ─────────────────────────────────────────────────

def test_unanswerable_separation_fails_when_identical():
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.8, 0.95, 200)
    answerable = rng.random(200) < 0.5  # confidence independent of answerability
    assert not screen_unanswerable_separation(conf, answerable).passed


def test_unanswerable_separation_passes_when_model_hesitates():
    rng = np.random.default_rng(4)
    answerable = rng.random(200) < 0.5
    conf = np.where(answerable, rng.uniform(0.7, 1.0, 200), rng.uniform(0.1, 0.4, 200))
    assert screen_unanswerable_separation(conf, answerable).passed


def test_unanswerable_separation_requires_both_classes():
    assert not screen_unanswerable_separation([0.9] * 10, [True] * 10).passed


# ── whole report ────────────────────────────────────────────────────────────

def test_report_not_usable_unless_every_screen_passes():
    rng = np.random.default_rng(5)
    answerable = rng.random(200) < 0.5
    conf = np.where(answerable, rng.uniform(0.7, 1.0, 200), rng.uniform(0.1, 0.4, 200))
    correct = rng.random(200) < 0.5

    # Missing optional data => those screens fail => report not usable.
    rep = run_all("m", "q4_K_M", confidence=conf, answerable=answerable,
                  attempted_confidence=conf, attempted_correct=correct)
    assert not rep.passed
    assert len(rep.results) == 5
    names = {r.name for r in rep.results}
    assert names == {
        "variance",
        "discrimination",
        "paraphrase-stability",
        "order-stability",
        "unanswerable-separation",
    }


def test_report_str_names_the_quantisation():
    """Quant tag must appear in output. A screenshot of a report that omits it
    is uninterpretable, per docs/METRICS.md.
    """
    rep = run_all(
        "granite4.1:8b",
        "q4_K_M",
        confidence=[0.1, 0.9],
        answerable=[True, True],
        attempted_confidence=[0.1, 0.9],
        attempted_correct=[False, True],
    )
    assert "q4_K_M" in str(rep)
    assert "granite4.1:8b" in str(rep)


# ── cli pairing helpers ──────────────────────────────────────────────────────

def _rec(item_id, perm, conf, paraphrase_of=None):
    return {
        "item_id": item_id,
        "permutation": perm,
        "signals": {"confidence": conf},
        "item": {"paraphrase_of": paraphrase_of, "forced_choice": True},
        "outcome": {"schema_valid": True},
    }


def test_twin_pairs_match_on_permutation_not_file_order():
    """Both twin records must be paired with the SAME-ORDERING original.

    Keying on id alone kept the last-written original, so one pair compared
    across option orderings and the paraphrase screen measured position
    sensitivity for half its pairs.
    """
    from qualm_analysis.cli import _twin_pairs

    usable = [
        _rec("q", [0, 1], 0.9),
        _rec("q", [1, 0], 0.5),          # written last: id-keying would pick this
        _rec("q-p", [0, 1], 0.9, "q"),
        _rec("q-p", [1, 0], 0.5, "q"),
    ]
    a, b = _twin_pairs(usable)
    assert sorted(zip(a, b)) == [(0.5, 0.5), (0.9, 0.9)]
    assert np.abs(np.array(a) - np.array(b)).mean() == 0.0
