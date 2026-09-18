"""Scoring for qualm run logs.

The headline metric is Type-2 AUROC, and the reason is not aesthetic. Type-2
AUROC is criterion-free: it depends only on the ORDERING of confidence values,
not their absolute level. That is what makes it survive quantisation, whereas
M-ratio profiles decorrelate entirely between quant levels. We run q4_K_M, so a
metric that moves when you requantise would be measuring the quantiser.

See docs/METRICS.md before changing anything here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A number that knows how uncertain it is.

    Point estimates on a few hundred items invite over-reading, which is this
    subfield's characteristic failure, so the CI travels with the value rather
    than being reported separately and lost.

    `n` is the number of RECORDS; `n_clusters` is the number of independent
    units the CI was actually computed over (items, once permutation records
    and paraphrase twins are grouped). The two are printed together because a
    run with `--permutations 2` reports n=116 for 58 items, and the record
    count is the number that gets over-read.
    """

    value: float
    lo: float
    hi: float
    n: int
    n_clusters: int | None = None

    def __str__(self) -> str:
        eff = "" if self.n_clusters is None else f", {self.n_clusters} clusters"
        return f"{self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n}{eff})"


def type2_auroc(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """P(confidence of a random correct item > that of a random incorrect one).

    0.5 = confidence carries no information about correctness.
    <0.5 = confidence is ANTI-correlated with correctness. That is worse than
           useless and a real, reportable outcome — do not clamp it away.

    Computed via the Mann-Whitney U identity rather than by sweeping thresholds,
    because the rank formulation handles ties exactly. Ties matter a great deal
    here: models cluster confidence on a few coarse values (0.85, 0.9, 0.95), so
    a tie-naive implementation is measurably wrong on real data rather than
    theoretically wrong.
    """
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    if conf.shape != corr.shape:
        raise ValueError("confidence and correct must be the same length")

    pos = conf[corr]
    neg = conf[~corr]
    if pos.size == 0 or neg.size == 0:
        # Undefined rather than 0.5: with only one class present there is
        # nothing to discriminate. Returning 0.5 would silently look like
        # "measured, no signal" when the truth is "not measurable".
        return float("nan")

    # Average ranks over the pooled sample; ties get their mean rank.
    ranks = _average_ranks(np.concatenate([pos, neg]))
    rank_sum_pos = ranks[: pos.size].sum()
    u = rank_sum_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks, ties averaged."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=float)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def bootstrap(
    fn,
    *arrays: Sequence,
    clusters: Sequence | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for any statistic over paired arrays.

    Resamples CLUSTERS, not rows, when `clusters` is given. Two kinds of row in
    a run log are not independent observations: the same item under k option
    orderings (`--permutations k`), and a paraphrase twin of an item. Drawing
    rows i.i.d. treats a 58-item run as 116 observations and reports a CI that
    is too narrow — every CI published before this argument existed was.
    Pass one cluster label per row (item id, with twins mapped to their
    original); whole clusters are drawn with replacement and their rows pooled.

    Without `clusters` this is the plain row bootstrap, kept for tests and for
    sets that genuinely have one row per item.
    """
    cols = [np.asarray(a) for a in arrays]
    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise ValueError("all arrays must be the same length")

    rng = np.random.default_rng(seed)
    point = fn(*cols)
    draws = np.empty(n_boot, dtype=float)

    if clusters is None:
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            draws[b] = fn(*(c[idx] for c in cols))
        n_clusters = None
    else:
        labels = np.asarray(clusters)
        if labels.shape[0] != n:
            raise ValueError("clusters must have one label per row")
        _, inverse = np.unique(labels, return_inverse=True)
        n_clusters = int(inverse.max()) + 1
        members = [np.flatnonzero(inverse == k) for k in range(n_clusters)]
        for b in range(n_boot):
            drawn = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([members[k] for k in drawn])
            draws[b] = fn(*(c[idx] for c in cols))

    good = draws[~np.isnan(draws)]
    if good.size == 0:
        return Estimate(float(point), float("nan"), float("nan"), n, n_clusters)
    lo, hi = np.quantile(good, [alpha / 2, 1 - alpha / 2])
    return Estimate(float(point), float(lo), float(hi), n, n_clusters)


def ece(confidence: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error.

    Reported, never headlined. ECE answers "does 0.9 mean 90%?", which is
    exactly the property quantisation disturbs — so it is useful for setting a
    gate threshold in real units and invalid for cross-quant comparison.
    Prefer a reliability diagram for presentation: one scalar hides whether a
    model is uniformly overconfident or badly wrong in a single bin.
    """
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for k in range(bins):
        # Half-open bins, last one closed, so conf == 1.0 is not dropped.
        sel = (conf >= edges[k]) & (conf < edges[k + 1]) if k < bins - 1 else (conf >= edges[k])
        if not sel.any():
            continue
        total += sel.mean() * abs(corr[sel].mean() - conf[sel].mean())
    return float(total)


@dataclass(frozen=True)
class AbstentionScores:
    """Abstention is a different question from calibration, and it is the
    product question. An abstention on an unanswerable item is a SUCCESS, which
    is why correctness alone cannot score this.
    """

    recall: float
    precision: float
    over_abstention: float
    n_unanswerable: int
    n_answerable: int


def abstention(answerable: Sequence[bool], abstained: Sequence[bool]) -> AbstentionScores:
    ans = np.asarray(answerable, dtype=bool)
    abst = np.asarray(abstained, dtype=bool)

    should = ~ans
    recall = float(abst[should].mean()) if should.any() else float("nan")
    precision = float(should[abst].mean()) if abst.any() else float("nan")
    # The failure that makes a product useless: declining work it could do.
    over = float(abst[ans].mean()) if ans.any() else float("nan")
    return AbstentionScores(recall, precision, over, int(should.sum()), int(ans.sum()))


@dataclass(frozen=True)
class OperatingPoint:
    """One gate a harness could actually set: accept iff confidence >= threshold."""

    threshold: float
    coverage: float
    accuracy: float
    n_retained: int


def operating_points(
    confidence: Sequence[float], correct: Sequence[bool]
) -> list[OperatingPoint]:
    """Coverage and retained accuracy at every threshold that EXISTS.

    Not a percentile curve. The previous version sliced the confidence-sorted
    records at 20/40/60/80% and reported accuracy on the slice — but these
    models emit three or four distinct confidence values, so a slice boundary
    almost always falls inside a tie group, and which tied records land above
    the cut is decided by file order (i.e. by which concurrent request finished
    first). The published "20% coverage → 0.941" row ranged 0.765–1.000 under
    record shuffling, and no confidence threshold could reach 20% coverage at
    all. A gate is a threshold, so report thresholds.

    Rows are in descending threshold order; the last row is the ungated
    baseline (threshold = min confidence, coverage 1.0).
    """
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    if conf.size == 0:
        return []
    out = []
    n = conf.size
    for t in sorted(set(conf.tolist()), reverse=True):
        sel = conf >= t
        out.append(
            OperatingPoint(
                float(t), float(sel.mean()), float(corr[sel].mean()), int(sel.sum())
            )
        )
    assert out[-1].n_retained == n
    return out


def load_run_log(path: str | Path) -> list[dict]:
    """Read a JSONL run log, skipping blank lines.

    Deliberately returns plain dicts. The run log is the archival artifact and
    its schema will grow; parsing into a rigid class here would mean old logs
    stop loading, which defeats the point of keeping them.
    """
    records = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def require_uniform_quant(records: Iterable[dict]) -> str:
    """Fail loudly if a set of records mixes quantisation levels.

    Not a nicety. Calibration metrics are not comparable across quant levels —
    M-ratio profiles decorrelate entirely — so silently pooling two quants
    produces a number that means nothing while looking perfectly reasonable.
    """
    quants = {r.get("model", {}).get("quantisation") for r in records}
    if None in quants:
        raise ValueError("run log contains records with no quantisation tag; cannot score")
    if len(quants) != 1:
        raise ValueError(f"refusing to pool across quantisations: {sorted(quants)}")
    return quants.pop()
