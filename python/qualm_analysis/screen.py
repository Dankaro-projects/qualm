"""Validity screening — runs BEFORE any profile is interpreted.

There is a literature arguing that benchmark-based confidence signals must be
screened for validity before their profiles mean anything, and a competing line
arguing LLMs have no internal confidence monitor at all — that "I'm not sure" is
a linguistic construction. Those two worlds produce similar-looking numbers.

This module is what distinguishes them. A model that fails here is REPORTED as
failing, not quietly dropped: given the state of the field, a null result is a
real contribution.

See docs/METRICS.md and docs/ROADMAP.md (M1 is a kill gate).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .metrics import Estimate, bootstrap, type2_auroc


@dataclass
class ScreenResult:
    name: str
    passed: bool
    detail: str
    value: float | None = None

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        v = "" if self.value is None else f" ({self.value:.3f})"
        return f"[{mark}] {self.name}{v} — {self.detail}"


@dataclass
class ScreenReport:
    model: str
    quantisation: str
    results: list[ScreenResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """All five must pass. Any failure makes downstream profiles
        uninterpretable, so this is deliberately not a score to be traded off.
        """
        return bool(self.results) and all(r.passed for r in self.results)

    def __str__(self) -> str:
        head = f"{self.model} @ {self.quantisation}: {'USABLE' if self.passed else 'NOT USABLE'}"
        return "\n".join([head, *(f"  {r}" for r in self.results)])


# Screen 1 ─ variance ────────────────────────────────────────────────────────
# Our own baseline: 36/36 confidence values inside [0.73, 0.98]. A field that
# barely moves cannot discriminate anything, yet still yields an AUROC near 0.5
# that reads as "weak signal" rather than "no measurement". Catch it explicitly.

def screen_variance(
    confidence: Sequence[float], min_sd: float = 0.05, min_distinct: int = 4
) -> ScreenResult:
    conf = np.asarray(confidence, dtype=float)
    sd = float(conf.std())
    distinct = int(np.unique(conf).size)
    ok = sd >= min_sd and distinct >= min_distinct
    return ScreenResult(
        "variance",
        ok,
        f"sd={sd:.3f} (need >={min_sd}), {distinct} distinct values "
        f"(need >={min_distinct}), range [{conf.min():.2f}, {conf.max():.2f}]",
        sd,
    )


# Screen 2 ─ discrimination above chance ─────────────────────────────────────
# Bootstrap CI, not a point estimate. A point AUROC of 0.56 on 200 items is
# entirely consistent with no signal.

def screen_discrimination(
    confidence: Sequence[float],
    correct: Sequence[bool],
    clusters: Sequence | None = None,
    seed: int = 0,
) -> ScreenResult:
    est: Estimate = bootstrap(
        type2_auroc, confidence, correct, clusters=clusters, seed=seed
    )
    ok = bool(est.lo > 0.5)
    return ScreenResult(
        "discrimination",
        ok,
        f"Type-2 AUROC {est} — CI lower bound must exceed 0.5",
        est.value,
    )


# Screen 3 ─ stability under paraphrase ──────────────────────────────────────
# Confidence that swings when a question is reworded is measuring surface form.

def screen_paraphrase_stability(
    original_conf: Sequence[float],
    twin_conf: Sequence[float],
    max_mean_abs_delta: float = 0.15,
) -> ScreenResult:
    a = np.asarray(original_conf, dtype=float)
    b = np.asarray(twin_conf, dtype=float)
    if a.size == 0:
        return ScreenResult("paraphrase-stability", False, "no paraphrase twins in set")
    delta = float(np.abs(a - b).mean())
    ok = delta <= max_mean_abs_delta
    return ScreenResult(
        "paraphrase-stability",
        ok,
        f"mean |Δconfidence| = {delta:.3f} across {a.size} twins "
        f"(need <={max_mean_abs_delta})",
        delta,
    )


# Screen 4 ─ stability under option reordering ───────────────────────────────
# Position sensitivity is a well-documented confound in multiple-choice items.

def screen_order_stability(
    conf_by_permutation: Sequence[Sequence[float]],
    max_mean_sd: float = 0.10,
) -> ScreenResult:
    """conf_by_permutation: one row per item, one column per option ordering."""
    m = np.asarray(conf_by_permutation, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        return ScreenResult("order-stability", False, "need >=2 orderings per item")
    per_item_sd = m.std(axis=1)
    mean_sd = float(per_item_sd.mean())
    ok = mean_sd <= max_mean_sd
    return ScreenResult(
        "order-stability",
        ok,
        f"mean within-item sd across orderings = {mean_sd:.3f} "
        f"over {m.shape[0]} items (need <={max_mean_sd})",
        mean_sd,
    )


# Screen 5 ─ answerable vs unanswerable separation ───────────────────────────
# If confidence is identical on answerable and unanswerable items, there is no
# hesitation present to measure and the rest of the pipeline is moot.

def screen_unanswerable_separation(
    confidence: Sequence[float],
    answerable: Sequence[bool],
    min_gap: float = 0.05,
    clusters: Sequence | None = None,
    seed: int = 0,
) -> ScreenResult:
    conf = np.asarray(confidence, dtype=float)
    ans = np.asarray(answerable, dtype=bool)
    if not ans.any() or not (~ans).any():
        return ScreenResult(
            "unanswerable-separation", False, "set lacks both answerable and unanswerable items"
        )

    def gap(c: np.ndarray, a: np.ndarray) -> float:
        if not a.any() or not (~a).any():
            return float("nan")
        return float(c[a].mean() - c[~a].mean())

    est = bootstrap(gap, conf, ans, clusters=clusters, seed=seed)
    # One-sided: confidence should be HIGHER on answerable items.
    ok = bool(est.lo > min_gap)
    return ScreenResult(
        "unanswerable-separation",
        ok,
        f"mean confidence gap (answerable − unanswerable) = {est} "
        f"— CI lower bound must exceed {min_gap}",
        est.value,
    )


def run_all(
    model: str,
    quantisation: str,
    *,
    confidence: Sequence[float],
    answerable: Sequence[bool],
    attempted_confidence: Sequence[float],
    attempted_correct: Sequence[bool],
    paraphrase_pairs: tuple[Sequence[float], Sequence[float]] | None = None,
    order_permutations: Sequence[Sequence[float]] | None = None,
    clusters: Sequence | None = None,
    attempted_clusters: Sequence | None = None,
    seed: int = 0,
) -> ScreenReport:
    """Run every screen.

    Takes TWO confidence arrays, deliberately, because the screens need
    different populations and conflating them is a real error:

      - `confidence` / `answerable` cover EVERY probe, including abstentions.
        Variance and answerable-separation are about the full distribution; an
        abstention still carries a confidence value and excluding those would
        hide the phenomenon.
      - `attempted_confidence` / `attempted_correct` cover only probes the model
        actually answered. Discrimination needs a correctness label, and an
        abstention has none — feeding it a padded array silently misaligns
        confidence against correctness.

    Missing optional data FAILS its screen rather than skipping it. An unrun
    screen is not a passed screen, and treating it as one is how a profile gets
    published on an unvalidated signal.

    `clusters` / `attempted_clusters` are per-row cluster labels for the two
    populations (item id, twins mapped to their original), so the bootstrap
    CIs behind screens 2 and 5 resample items rather than permutation records.
    """
    report = ScreenReport(model=model, quantisation=quantisation)
    report.results.append(screen_variance(confidence))
    report.results.append(
        screen_discrimination(
            attempted_confidence, attempted_correct, clusters=attempted_clusters, seed=seed
        )
    )

    if paraphrase_pairs is None:
        report.results.append(
            ScreenResult("paraphrase-stability", False, "not run — no twins provided")
        )
    else:
        report.results.append(screen_paraphrase_stability(*paraphrase_pairs))

    if order_permutations is None:
        report.results.append(
            ScreenResult("order-stability", False, "not run — no permutations provided")
        )
    else:
        report.results.append(screen_order_stability(order_permutations))

    report.results.append(
        screen_unanswerable_separation(confidence, answerable, clusters=clusters, seed=seed)
    )
    return report
