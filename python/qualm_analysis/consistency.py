"""Self-consistency across seeded runs — the signal a gate should probably use.

Stated confidence on the passage-QA task was flat (1.0 on 98% of items) and
blind to the one error that matters there — answering from a passage that
does not contain the answer (arXiv 2607.08456 predicts exactly this). The
alternative signal is agreement: ask the same item N times at temperature > 0
with different seeds and see whether the model gives the SAME answer. arXiv
2604.24070 measured agreement at AUROC 0.999 against verbalised 0.554 on a
small local model. This module measures it here, on task-shaped items, and
turns it into the routing table a gate needs.

  uv run python -m qualm_analysis.consistency \\
      --base results/raw/gemma4-12b-squad2-r1.jsonl \\
      --samples results/raw/gemma4-12b-squad2-s1.jsonl ... \\
      [--accept-at 1.0]

`--base` is the deterministic run (temp 0, seed 0) — what runs in production
when samples=1. `--samples` are the seeded runs. Records are joined on
(item_id, permutation).

Task-correct, the metric for a workflow step: an answer that matches gold, OR
a decline (abstain / no answer) on an unanswerable item. A confident answer
to an unanswerable item is the error the gate exists to stop.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .metrics import bootstrap, load_run_log, operating_points, type2_auroc


def norm(s: str | None) -> str:
    """Mirror of item.rs::norm."""
    if s is None:
        return ""
    s = " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())
    while True:
        for a in ("the ", "an ", "a "):
            if s.startswith(a):
                s = s[len(a):]
                break
        else:
            return s


def _key(r: dict) -> tuple:
    return (r["item_id"], tuple(r.get("permutation") or ()))


def _vote(r: dict) -> str:
    """What the record 'said': the normalised answer, or DECLINE if it
    abstained, produced no answer, or was schema-invalid (a no-answer reply
    that the harness rejected). Forced-choice records vote by canonical
    option index so 'same option' is compared, not 'same letter'."""
    o = r.get("outcome", {})
    if not o.get("schema_valid") or o.get("abstained"):
        return "DECLINE"
    if o.get("choice"):
        perm = r.get("permutation") or []
        idx = ord(o["choice"].strip().upper()[0]) - 65
        return f"opt:{perm[idx] if idx < len(perm) else idx}"
    a = o.get("raw_answer")
    return norm(a) if a and norm(a) else "DECLINE"


def _task_correct(r: dict, vote: str) -> bool:
    if vote == "DECLINE":
        return not r["item"]["answerable"]
    return bool(r.get("outcome", {}).get("correct"))


def analyse(base_path: str, sample_paths: list[str], accept_at: float = 1.0) -> dict:
    base = {_key(r): r for r in load_run_log(base_path)}
    samples: dict[tuple, list[dict]] = defaultdict(list)
    for p in sample_paths:
        for r in load_run_log(p):
            samples[_key(r)].append(r)

    rows = []
    for k, b in base.items():
        votes = [_vote(b)] + [_vote(s) for s in samples.get(k, [])]
        maj, n_maj = Counter(votes).most_common(1)[0]
        agreement = n_maj / len(votes)
        b_vote = votes[0]
        rows.append({
            "key": k,
            "answerable": bool(b["item"]["answerable"]),
            "category": b["item"]["category"],
            "base_vote": b_vote,
            "base_declined": b_vote == "DECLINE",
            "base_task_correct": _task_correct(b, b_vote),
            "base_conf": (b.get("signals") or {}).get("confidence"),
            "agreement": agreement,
            "unanimous": agreement >= 1.0,
            "n_votes": len(votes),
        })
    return {"rows": rows, "n_samples": max((len(v) for v in samples.values()), default=0), "accept_at": accept_at}


def report(res: dict) -> None:
    rows = res["rows"]
    n = len(rows)
    acc_at = res["accept_at"]
    ans = np.array([r["answerable"] for r in rows])
    declined = np.array([r["base_declined"] for r in rows])
    tc = np.array([r["base_task_correct"] for r in rows])
    conf = np.array([r["base_conf"] if r["base_conf"] is not None else 0.0 for r in rows], dtype=float)
    agree = np.array([r["agreement"] for r in rows], dtype=float)
    unan = np.array([r["unanimous"] for r in rows])

    print("=" * 74)
    print(f"SELF-CONSISTENCY  base + {res['n_samples']} seeded samples · {n} items "
          f"({int(ans.sum())} answerable, {int((~ans).sum())} unanswerable)")
    print("=" * 74)
    print(f"\nUNGATED task accuracy (take the base output as-is): {tc.mean():.3f}   "
          f"errors {int((~tc).sum())}  = {int((~tc & ans).sum())} wrong answers + "
          f"{int((~tc & ~ans).sum())} answered-unanswerable + {int((~tc & declined & ans).sum())} wrongly declined")

    # Gates route ANSWERED items; declines are already a decision.
    answered = ~declined

    def gate_line(name, accept):
        acc = answered & accept
        rev = answered & ~accept
        print(f"  {name:38s} accept {acc.mean():5.0%} of items · accepted accuracy "
              f"{tc[acc].mean() if acc.any() else float('nan'):.3f} · errors let through {int((~tc & acc).sum()):3d} · "
              f"to review {int(rev.sum()):3d} (of which {int((~tc & rev).sum())} would have been errors)")

    print("\nGATES on answered items  (declined items pass straight through as declines)")
    gate_line("none (accept everything answered)", np.ones(n, bool))
    gate_line(f"stated confidence >= {acc_at}", conf >= acc_at)
    gate_line("unanimous across samples", unan)
    gate_line(f"both (conf >= {acc_at} AND unanimous)", (conf >= acc_at) & unan)

    # Where the errors live and whether either signal sees them.
    print("\nSIGNAL vs the two error kinds  (answered items only)")
    wrong_ans = answered & ans & ~tc
    halluc = answered & ~ans
    right = answered & ans & tc
    for name, sel in [("correct answers", right), ("wrong answers (answerable)", wrong_ans),
                      ("answered-unanswerable", halluc)]:
        if sel.any():
            print(f"  {name:28s} n={int(sel.sum()):3d}  mean conf {conf[sel].mean():.3f}  "
                  f"mean agreement {agree[sel].mean():.3f}  unanimous {unan[sel].mean():.0%}")

    # AUROCs, item level: does the signal rank task-correct above task-wrong
    # among answered items? Answerable-only (clean construct) and all answered
    # (the routing question, which legitimately includes the unanswerable half
    # because on THIS task answering an unanswerable IS the error).
    print("\nAUROC (item level, answered items)")
    for name, sel in [("answerable-only", answered & ans), ("all answered", answered)]:
        y = tc[sel]
        if y.all() or not y.any():
            print(f"  {name:18s} single class")
            continue
        a1 = bootstrap(type2_auroc, agree[sel], y)
        a2 = bootstrap(type2_auroc, conf[sel], y)
        print(f"  {name:18s} agreement {a1}   verbalised {a2}   [{int((~y).sum())} errors]")

    print("\nOPERATING POINTS on agreement (all answered items, task-correct)")
    for op in operating_points(agree[answered], tc[answered]):
        print(f"  agreement >= {op.threshold:<5.3f} coverage {op.coverage:.0%}   accuracy {op.accuracy:.3f}   (n={op.n_retained})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--samples", nargs="+", required=True, type=Path)
    ap.add_argument("--accept-at", type=float, default=1.0)
    args = ap.parse_args(argv)
    report(analyse(str(args.base), [str(p) for p in args.samples], args.accept_at))
    return 0


if __name__ == "__main__":
    sys.exit(main())
