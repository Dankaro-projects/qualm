"""Score a run log: validity screen first, then metrics.

Screen before metrics is the order enforced here, not a suggestion. If the
signal does not pass, the metrics below it are numbers about noise — and they
will look perfectly reasonable, which is the danger.

  uv run python -m qualm_analysis.cli results/raw/*.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .metrics import (
    Estimate,
    abstention,
    bootstrap,
    ece,
    load_run_log,
    operating_points,
    require_uniform_quant,
    type2_auroc,
)
from .screen import run_all


def _extract(records: list[dict]):
    """Pull the arrays the metrics need, dropping unusable rows loudly.

    A row with schema_valid=false is a harness/format failure, not evidence
    about metacognition. Silently including it would let a formatting problem
    masquerade as poor calibration.
    """
    usable, dropped = [], 0
    for r in records:
        o = r.get("outcome", {})
        s = r.get("signals") or {}
        if not o.get("schema_valid") or s.get("confidence") is None:
            dropped += 1
            continue
        usable.append(r)
    return usable, dropped


def _cluster_key(r: dict) -> str:
    """The independent unit behind a record: the item, with a paraphrase twin
    folded into its original. Permutation records of one item share it."""
    return r["item"].get("paraphrase_of") or r["item_id"]


def report(path: Path) -> dict:
    records = load_run_log(path)
    quant = require_uniform_quant(records)
    model = records[0]["model"]["name"]
    usable, dropped = _extract(records)

    conf = np.array([r["signals"]["confidence"] for r in usable], dtype=float)
    answerable = np.array([r["item"]["answerable"] for r in usable], dtype=bool)
    abstained = np.array([r["outcome"]["abstained"] for r in usable], dtype=bool)
    challenged = np.array(
        [bool(r["outcome"].get("premise_challenged")) for r in usable], dtype=bool
    )
    clusters = np.array([_cluster_key(r) for r in usable])
    # `correct` is None for a correct abstention on an unanswerable item — that
    # is a success with no accuracy label. Treated as its own case, never as
    # False, which would punish the behaviour we are trying to measure. Since
    # 0.3.0 the same holds for a premise rejection on an unanswerable item.
    correct_raw = [r["outcome"]["correct"] for r in usable]

    print("=" * 74)
    print(
        f"{model} @ {quant}    n={len(usable)} usable, {dropped} dropped, "
        f"{len(set(clusters))} independent items"
    )
    print("=" * 74)

    # ── populations ────────────────────────────────────────────────────────
    # Attempted = NOT ABSTAINED and carrying a correctness label. Previously
    # `correct is not None` alone, which leaked abstentions on ANSWERABLE items
    # into the AUROC (they carry correct=False), feeding it the most favourable
    # possible negative at minimum confidence.
    attempted = ~abstained & np.array([c is not None for c in correct_raw], dtype=bool)
    att_conf = conf[attempted]
    att_correct = np.array([c for c, a in zip(correct_raw, attempted) if a], dtype=bool)
    att_answerable = answerable[attempted]
    att_clusters = clusters[attempted]
    n_err = int((~att_correct).sum()) if att_correct.size else 0

    # The CLEAN construct: answerable items only. An attempted unanswerable
    # item is wrong by construction, so pooling it into the negative class
    # measures answerability separation (screen 5) under the AUROC's name.
    # METRICS.md has said so since the M1 correction; the M3 headline quoted
    # the pooled figure anyway. Both are reported, labelled, from here on.
    ans_conf = att_conf[att_answerable]
    ans_correct = att_correct[att_answerable]
    ans_clusters = att_clusters[att_answerable]
    n_ans_err = int((~ans_correct).sum()) if ans_correct.size else 0
    n_ans_err_items = len(set(ans_clusters[~ans_correct])) if ans_correct.size else 0

    twins = _twin_pairs(usable)
    perms = _order_permutations(usable)
    rep = run_all(
        model,
        quant,
        confidence=conf,
        answerable=answerable,
        attempted_confidence=ans_conf,
        attempted_correct=ans_correct,
        paraphrase_pairs=twins,
        order_permutations=perms,
        clusters=clusters,
        attempted_clusters=ans_clusters,
    )
    print("\nVALIDITY SCREEN  (discrimination on the answerable-only construct)")
    for r in rep.results:
        print(f"  {r}")
    print(f"  => {'USABLE' if rep.passed else 'NOT USABLE'}")
    # Screen 5 runs on the full set by design (an abstention still carries a
    # confidence). But one model writes 0.0 on every abstention, which alone
    # produces a large gap. The attempted-only gap is printed so that a PASS
    # driven entirely by the abstention convention is visible as such.
    gap_att = _gap_attempted(att_conf, att_answerable, att_clusters)
    if gap_att is not None:
        print(f"     attempted-only gap (caveat, not the verdict): {gap_att}")

    # ── metrics ────────────────────────────────────────────────────────────
    print("\nTYPE-2 AUROC")
    auroc_ans: Estimate = bootstrap(
        type2_auroc, ans_conf, ans_correct, clusters=ans_clusters
    )
    auroc_pool: Estimate = bootstrap(
        type2_auroc, att_conf, att_correct, clusters=att_clusters
    )
    # The error count is printed adjacent to the AUROC, always. An empty or
    # tiny negative class makes the metric undefined or wildly imprecise while
    # the point estimate still looks authoritative — that is how the open-ended
    # pilot published 0.901 for a construct it could not measure.
    print(
        f"  answerable-only  {auroc_ans}   "
        f"[negative class: {n_ans_err} error record(s) on {n_ans_err_items} item(s)]"
    )
    print(
        f"  pooled           {auroc_pool}   "
        f"[negative class: {n_err} = {n_ans_err} answerable + "
        f"{n_err - n_ans_err} attempted-unanswerable]"
    )
    if n_ans_err == 0:
        print("  !! answerable-only AUROC IS UNDEFINED — no errors, nothing to discriminate")
    elif n_ans_err_items < 5:
        print(f"  !! only {n_ans_err_items} item(s) in error: treat the CI as indicative, not tight")
    print(
        f"  ECE (answerable) {ece(ans_conf, ans_correct):.3f}   "
        f"[{quant} — not comparable across quant levels]"
    )
    print(
        f"  accuracy         {ans_correct.mean():.3f} on {ans_correct.size} attempted answerable"
        f"  ·  {att_correct.mean():.3f} on {att_correct.size} attempted overall"
    )

    # Declined = abstained OR rejected the premise. Both are the model refusing
    # the question as posed; the two are reported separately below because a
    # premise rejection is the more specific (and more useful) behaviour.
    declined = abstained | challenged
    ab = abstention(answerable, declined)
    print("\nDECLINING  (abstain ∪ premise-rejection)")
    print(f"  recall          {ab.recall:.3f}  (of {ab.n_unanswerable} unanswerable, declined)")
    print(f"  precision       {ab.precision:.3f}")
    print(f"  over-declining  {ab.over_abstention:.3f}  (of {ab.n_answerable} answerable, wrongly declined)")

    print("\nOPERATING POINTS  (answerable items; accept iff confidence >= threshold)")
    for op in operating_points(ans_conf, ans_correct):
        print(
            f"  conf >= {op.threshold:<6.4g} coverage {op.coverage:.0%}   "
            f"retained accuracy {op.accuracy:.3f}   (n={op.n_retained})"
        )

    # ── self-consistency vs verbalised confidence, item level ──────────────
    sc = _self_consistency(usable)
    if sc is not None:
        agree, vconf, mcorrect, items_c = sc
        print(
            f"\nSELF-CONSISTENCY  (item level, answerable forced-choice items with >=2 orderings; "
            f"n={len(items_c)} items, {int((~mcorrect).sum())} wrong by majority)"
        )
        if mcorrect.all() or not mcorrect.any():
            print("  !! single class — nothing to discriminate at item level")
        else:
            a_sc = bootstrap(type2_auroc, agree, mcorrect, clusters=items_c)
            a_vc = bootstrap(type2_auroc, vconf, mcorrect, clusters=items_c)
            print(f"  AUROC  agreement→majority-correct   {a_sc}")
            print(f"  AUROC  verbalised→majority-correct  {a_vc}")
            print("  gates on agreement:")
            for op in operating_points(agree, mcorrect):
                print(
                    f"    agreement >= {op.threshold:<5.3f} coverage {op.coverage:.0%}   "
                    f"retained accuracy {op.accuracy:.3f}   (n={op.n_retained})"
                )
            both = (agree >= 1.0) & (vconf >= vconf.max())
            if both.any():
                print(
                    f"  combined gate (unanimous AND stated conf == {vconf.max():.2f}): "
                    f"coverage {both.mean():.0%}, retained accuracy {mcorrect[both].mean():.3f} "
                    f"(n={int(both.sum())}) — vs {mcorrect.mean():.3f} ungated"
                )

    # ── the coherence probe ────────────────────────────────────────────────
    if challenged.any():
        by = Counter(r["item"]["category"] for r, c in zip(usable, challenged) if c)
        print(f"\nPREMISE REJECTED ({int(challenged.sum())} of {len(usable)} probes)")
        for c, k in sorted(by.items()):
            tot = sum(1 for r in usable if r["item"]["category"] == c)
            print(f"  {c:24s} {k}/{tot} ({k/tot:.0%})")

    print("\nCONFIDENCE BY CATEGORY  (all rows; abstained rows carry the model's own convention)")
    by_cat: dict[str, list[float]] = defaultdict(list)
    ab_by_cat: dict[str, list[bool]] = defaultdict(list)
    for r, d in zip(usable, declined):
        c = r["item"]["category"]
        by_cat[c].append(r["signals"]["confidence"])
        ab_by_cat[c].append(bool(d))
    for cat in sorted(by_cat):
        v = np.array(by_cat[cat])
        a = np.array(ab_by_cat[cat])
        n_items = len({_cluster_key(r) for r in usable if r["item"]["category"] == cat})
        print(
            f"  {cat:24s} n={v.size:3d} ({n_items:2d} items)  conf {v.mean():.3f}±{v.std():.3f}  "
            f"declined {a.mean():.0%}"
        )

    return {
        "model": model,
        "quant": quant,
        "n": len(usable),
        "n_items": len(set(clusters)),
        "screen_passed": rep.passed,
        "auroc": auroc_ans,
        "auroc_pooled": auroc_pool,
        "n_ans_err_items": n_ans_err_items,
        "abstention": ab,
        "by_category": {k: float(np.mean(v)) for k, v in by_cat.items()},
        "abstain_by_category": {k: float(np.mean(v)) for k, v in ab_by_cat.items()},
    }


def _self_consistency(usable: list[dict]):
    """Per answerable forced-choice item: majority option across orderings,
    agreement rate, and whether the majority is correct.

    The option each record picked is resolved to its CANONICAL index through
    the recorded permutation, so agreement means "same option", not "same
    letter". Records that abstained vote for "abstain"; an item whose majority
    abstained is dropped here (it is an over-abstention, counted elsewhere).
    Verbalised confidence for the item is the mean stated confidence over the
    records that agree with the majority — the same value the harness gate
    returns, so what is measured is what runs.

    `signals.agreement` in the run record stays null on purpose: the harness
    records the raw material (one record per ordering) and this derives the
    aggregate, so a re-analysis can change the rule without a re-run.
    """
    by_item: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        if r["item"].get("forced_choice") and r["item"]["answerable"]:
            by_item[r["item_id"]].append(r)
    agree, vconf, mcorrect, items = [], [], [], []
    for item_id, recs in by_item.items():
        if len(recs) < 2:
            continue
        votes = []
        for r in recs:
            if r["outcome"]["abstained"] or not r["outcome"].get("choice"):
                votes.append(("abstain", None, r))
                continue
            perm = r.get("permutation") or []
            idx = ord(r["outcome"]["choice"].strip().upper()[0]) - 65
            canon = perm[idx] if idx < len(perm) else idx
            votes.append((canon, r["outcome"]["correct"], r))
        keys = [v[0] for v in votes]
        maj, n_maj = Counter(keys).most_common(1)[0]
        if maj == "abstain":
            continue
        agreeing = [v for v in votes if v[0] == maj]
        agree.append(n_maj / len(votes))
        vconf.append(float(np.mean([v[2]["signals"]["confidence"] for v in agreeing])))
        mcorrect.append(bool(agreeing[0][1]))
        items.append(_cluster_key(recs[0]))
    if not items:
        return None
    return (
        np.array(agree, dtype=float),
        np.array(vconf, dtype=float),
        np.array(mcorrect, dtype=bool),
        np.array(items),
    )


def _gap_attempted(att_conf, att_answerable, att_clusters) -> Estimate | None:
    if not att_answerable.any() or att_answerable.all():
        return None

    def gap(c, a):
        if not a.any() or a.all():
            return float("nan")
        return float(c[a].mean() - c[~a].mean())

    return bootstrap(gap, att_conf, att_answerable, clusters=att_clusters)


def _order_permutations(usable: list[dict]):
    """Confidence per item across option orderings, for validity screen #4.

    This screen could never run before, because no permutations existed. Only
    items with >=2 orderings are included; a ragged set is padded to the minimum
    common width rather than dropped, so a partial run still yields a verdict.
    """
    by_item: dict[str, list[float]] = defaultdict(list)
    for r in usable:
        if r["item"].get("forced_choice"):
            by_item[r["item_id"]].append(r["signals"]["confidence"])
    rows = [v for v in by_item.values() if len(v) >= 2]
    if not rows:
        return None
    width = min(len(v) for v in rows)
    return [v[:width] for v in rows]


def _twin_pairs(usable: list[dict]):
    """Match paraphrase twins to their originals, for validity screen #3.

    Keyed on (item id, permutation), not item id alone. Keying on id kept
    whichever original record was written last, so half the pairs compared a
    twin under one option ordering against its original under the other —
    the paraphrase screen was partly measuring order sensitivity, which is the
    confound the dataset generator had just been fixed to avoid.
    """
    by_key = {(r["item_id"], _perm_key(r)): r for r in usable}
    a, b = [], []
    for r in usable:
        orig = r["item"].get("paraphrase_of")
        if not orig:
            continue
        o = by_key.get((orig, _perm_key(r)))
        if o is not None:
            a.append(o["signals"]["confidence"])
            b.append(r["signals"]["confidence"])
    return (a, b) if a else None


def _perm_key(r: dict) -> tuple:
    p = r.get("permutation")
    return tuple(p) if p else ()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    args = ap.parse_args(argv)

    summaries = [report(p) for p in args.logs]

    if len(summaries) > 1:
        print("\n" + "=" * 74)
        print("LEADERBOARD  (Type-2 AUROC, answerable-only construct, cluster bootstrap)")
        print("=" * 74)
        print(
            f"{'model':16s} {'quant':7s} {'answerable-only':34s} {'pooled':34s} "
            f"{'screen':7s} {'decl.recall':>11s}"
        )
        for s in sorted(summaries, key=lambda x: -(x["auroc"].value or 0)):
            print(
                f"{s['model']:16s} {s['quant']:7s} {str(s['auroc']):34s} "
                f"{str(s['auroc_pooled']):34s} "
                f"{'PASS' if s['screen_passed'] else 'FAIL':7s} "
                f"{s['abstention'].recall:>11.3f}"
            )
        print("\nCoherence probe — mean confidence, answerable vs coherent-false-premise (all rows):")
        for s in summaries:
            ans = s["by_category"].get("answerable")
            cfp = s["by_category"].get("coherent-false-premise")
            obv = s["by_category"].get("false-premise")
            if ans and cfp:
                print(
                    f"  {s['model']:22s} answerable {ans:.3f} | "
                    f"coherent-FP {cfp:.3f} (Δ{ans - cfp:+.3f}) | "
                    f"obvious-FP {obv:.3f} (Δ{ans - obv:+.3f})"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
