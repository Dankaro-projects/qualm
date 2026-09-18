"""Tests against hand-computable fixtures.

Under this project's verification standard, a scoring function that runs is not a
scoring function that is correct. Every case here has an answer derivable on
paper, so a regression is a real failure rather than a changed number nobody
can adjudicate.
"""

import math

import numpy as np
import pytest

from qualm_analysis.metrics import (
    abstention,
    bootstrap,
    ece,
    operating_points,
    require_uniform_quant,
    type2_auroc,
)


def test_perfect_separation():
    # Every correct item above every incorrect one.
    assert type2_auroc([0.9, 0.8, 0.4, 0.3], [True, True, False, False]) == 1.0


def test_inverted_separation():
    # Confidence anti-correlated with correctness. Must report 0.0, not clamp:
    # a model whose confidence points the wrong way is a finding.
    assert type2_auroc([0.3, 0.4, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_chance():
    # Interleaved: 2 correct, 2 incorrect, perfectly alternating.
    # Pairs (correct, incorrect): (0.9,0.8)win (0.9,0.3)win (0.4,0.8)lose (0.4,0.3)win
    # => 3/4
    assert type2_auroc([0.9, 0.4, 0.8, 0.3], [True, True, False, False]) == 0.75


def test_all_ties_is_exactly_half():
    """The case that matters most on real data.

    Models cluster confidence on a few coarse values, so ties are the norm, not
    an edge case. With every value identical there is no ordering information
    and the answer must be exactly 0.5 — a tie-naive implementation drifts here.
    """
    assert type2_auroc([0.85] * 6, [True, True, True, False, False, False]) == 0.5


def test_partial_ties():
    # pos = [0.9, 0.5], neg = [0.9, 0.1]
    # pairs: (0.9 vs 0.9) tie = 0.5 | (0.9 vs 0.1) = 1 | (0.5 vs 0.9) = 0 | (0.5 vs 0.1) = 1
    # => (0.5 + 1 + 0 + 1) / 4 = 0.625
    assert type2_auroc([0.9, 0.5, 0.9, 0.1], [True, True, False, False]) == 0.625


def test_single_class_is_nan_not_half():
    """Undefined must not masquerade as 'measured, no signal'."""
    assert math.isnan(type2_auroc([0.9, 0.8], [True, True]))
    assert math.isnan(type2_auroc([0.9, 0.8], [False, False]))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        type2_auroc([0.9, 0.8], [True])


def test_ece_perfectly_calibrated_is_zero():
    # 10 items at conf 0.5, exactly half correct.
    conf = [0.5] * 10
    corr = [True] * 5 + [False] * 5
    assert ece(conf, corr, bins=10) == pytest.approx(0.0, abs=1e-12)


def test_ece_maximally_wrong_is_one():
    # Total confidence, total failure.
    assert ece([1.0] * 8, [False] * 8, bins=10) == pytest.approx(1.0)


def test_ece_includes_confidence_of_one():
    """conf == 1.0 must land in the last bin, not be dropped."""
    # 4 items at 1.0, all correct => perfectly calibrated => 0.
    assert ece([1.0] * 4, [True] * 4, bins=10) == pytest.approx(0.0)


def test_abstention_scores():
    # items:      A(ans) B(ans) C(unans) D(unans)
    answerable = [True, True, False, False]
    abstained = [False, True, True, False]
    s = abstention(answerable, abstained)
    assert s.recall == 0.5           # 1 of 2 unanswerable declined
    assert s.precision == 0.5        # 2 abstentions, 1 was right
    assert s.over_abstention == 0.5  # 1 of 2 answerable wrongly declined
    assert s.n_unanswerable == 2
    assert s.n_answerable == 2


def test_operating_points_are_thresholds_not_slices():
    conf = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5]
    corr = [True, True, True, True, False, False]
    ops = operating_points(conf, corr)
    assert [o.threshold for o in ops] == [0.99, 0.9, 0.8, 0.7, 0.6, 0.5]
    assert ops[0].accuracy == 1.0 and ops[0].coverage == pytest.approx(1 / 6)
    assert ops[-1].coverage == pytest.approx(1.0)    # ungated baseline
    assert ops[-1].accuracy == pytest.approx(4 / 6)  # overall accuracy


def test_operating_points_do_not_depend_on_record_order_under_ties():
    # The real pattern: most records tied at 1.0, a few wrong among them. A
    # percentile slice inside the tie group depended on file order; a
    # threshold does not.
    conf = [1.0] * 10 + [0.5] * 2
    corr = [True] * 8 + [False] * 2 + [False] * 2
    ref = operating_points(conf, corr)
    rng = np.random.default_rng(1)
    for _ in range(20):
        idx = rng.permutation(len(conf))
        ops = operating_points([conf[i] for i in idx], [corr[i] for i in idx])
        assert ops == ref
    assert [o.threshold for o in ref] == [1.0, 0.5]
    assert ref[0].coverage == pytest.approx(10 / 12)
    assert ref[0].accuracy == pytest.approx(0.8)


# ── cluster bootstrap ────────────────────────────────────────────────────────

def test_cluster_bootstrap_widens_ci_for_duplicated_records():
    """Two permutation records per item carry no more information than one.

    Fixture: 12 items, each duplicated (as `--permutations 2` does), with a
    confidence signal that separates correct from wrong imperfectly. The row
    bootstrap thinks it has 24 observations; the cluster bootstrap knows it
    has 12, and its CI must be wider.
    """
    n_items = 12
    correct_item = np.array([True] * 8 + [False] * 4)
    # Overlapping on purpose: AUROC must be strictly inside (0.5, 1) or both
    # intervals collapse to a point and the comparison is vacuous.
    conf_item = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.9, 0.6, 0.5, 0.4, 0.3])
    conf = np.repeat(conf_item, 2)
    corr = np.repeat(correct_item, 2)
    clusters = np.repeat(np.arange(n_items), 2)

    rows = bootstrap(type2_auroc, conf, corr, n_boot=1000)
    clus = bootstrap(type2_auroc, conf, corr, clusters=clusters, n_boot=1000)

    assert rows.value == pytest.approx(clus.value)   # same point estimate
    assert rows.n == clus.n == 24
    assert rows.n_clusters is None and clus.n_clusters == 12
    assert (clus.hi - clus.lo) > (rows.hi - rows.lo)  # the whole point


def test_cluster_bootstrap_equals_row_bootstrap_when_every_row_is_its_own_cluster():
    rng = np.random.default_rng(3)
    conf = rng.random(30)
    corr = conf + rng.normal(0, 0.3, 30) > 0.5
    a = bootstrap(type2_auroc, conf, corr, n_boot=500, seed=7)
    b = bootstrap(type2_auroc, conf, corr, clusters=np.arange(30), n_boot=500, seed=7)
    # Same draws are not guaranteed (different RNG call pattern), but the
    # intervals must agree to well within Monte-Carlo noise.
    assert a.value == pytest.approx(b.value)
    assert abs((a.hi - a.lo) - (b.hi - b.lo)) < 0.05


def test_cluster_bootstrap_rejects_mismatched_labels():
    with pytest.raises(ValueError):
        bootstrap(type2_auroc, [0.1, 0.9], [False, True], clusters=[0])


def test_require_uniform_quant_rejects_mixed():
    recs = [
        {"model": {"quantisation": "q4_K_M"}},
        {"model": {"quantisation": "q8_0"}},
    ]
    with pytest.raises(ValueError, match="refusing to pool"):
        require_uniform_quant(recs)


def test_require_uniform_quant_rejects_missing_tag():
    with pytest.raises(ValueError, match="no quantisation tag"):
        require_uniform_quant([{"model": {}}])


def test_require_uniform_quant_accepts_uniform():
    recs = [{"model": {"quantisation": "q4_K_M"}}] * 3
    assert require_uniform_quant(recs) == "q4_K_M"


def test_auroc_matches_sklearn_style_reference_on_random_data():
    """Cross-check the rank implementation against a brute-force pair count."""
    rng = np.random.default_rng(7)
    conf = rng.choice([0.1, 0.5, 0.85, 0.9, 1.0], size=60)  # deliberate ties
    corr = rng.random(60) < 0.5

    pos, neg = conf[corr], conf[~corr]
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in pos
        for n in neg
    )
    brute = wins / (pos.size * neg.size)
    assert type2_auroc(conf, corr) == pytest.approx(brute)
