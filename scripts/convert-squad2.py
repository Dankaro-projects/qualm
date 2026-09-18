#!/usr/bin/env python3
"""Build datasets/squad2-task.jsonl — the Door 2 task with checkable outputs.

"Answer the question from the passage; if the passage does not contain the
answer, decline." That is the shape of a professional document-QA workflow
step, and SQuAD 2.0 (rajpurkar/squad_v2, CC BY-SA 4.0) supplies ground truth
for both halves: gold answer spans for answerable questions and a built-in
unanswerable half. So a gate can be measured here — gated vs ungated accuracy
on the same inputs.

Scoring: set-contains against the gold answer variants — the answer contains a
gold on token boundaries after the harness's norm() (one direction). Exact match scored "The
Mongol army under Jani Beg" wrong against gold "Jani Beg" (a confident correct
answer as an error, the worst direction for AUROC); containment over-credits
list answers that include the gold. Closer to SQuAD's F1 than to its EM. Category for the
unanswerable half is 'missing-context'. Sampled with a fixed seed from the
validation split, contexts capped in length.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data-cache"
OUT = ROOT / "datasets" / "squad2-task.jsonl"
# The same items as question/passage/gold records, so that other tools can be
# evaluated on exactly the inputs the harness measured.
OUT_EVAL = ROOT / "datasets" / "squad2-task.eval.json"
SEED = 20260817
N_EACH = 150
MAX_CONTEXT_CHARS = 1200

TEMPLATE = (
    "Using ONLY the passage below, answer the question. If the passage does not "
    "contain the answer, decline.\n\nPassage: {ctx}\n\nQuestion: {q}"
)


def clean(s: str) -> str:
    return " ".join(str(s).split())


def main() -> int:
    p = CACHE / "squad_v2-validation.parquet"
    if not p.exists():
        df = pd.read_parquet(
            "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/squad_v2/validation-00000-of-00001.parquet"
        )
        df.to_parquet(p)
    df = pd.read_parquet(p)
    df = df[df["context"].str.len() <= MAX_CONTEXT_CHARS]
    rows = df.to_dict("records")
    ans = [r for r in rows if len(r["answers"]["text"]) > 0]
    una = [r for r in rows if len(r["answers"]["text"]) == 0]
    rng = random.Random(SEED)
    rng.shuffle(ans)
    rng.shuffle(una)
    items, eval_items = [], []
    for r in ans[:N_EACH]:
        golds = sorted({clean(t) for t in r["answers"]["text"]})
        eval_items.append({"id": f"sq-{r['id']}", "question": clean(r["question"]),
                           "passage": clean(r["context"]), "answerable": True, "gold": golds})
        items.append({
            "id": f"sq-{r['id']}",
            "prompt": TEMPLATE.format(ctx=clean(r["context"]), q=clean(r["question"])),
            "answerable": True,
            "category": "answerable",
            "domain": clean(r["title"]),
            "context": clean(r["context"]),
            "scoring": {"method": "set-contains", "answer": golds},
            "source": "SQuAD2",
        })
    for r in una[:N_EACH]:
        eval_items.append({"id": f"sq-{r['id']}", "question": clean(r["question"]),
                           "passage": clean(r["context"]), "answerable": False, "gold": []})
        items.append({
            "id": f"sq-{r['id']}",
            "prompt": TEMPLATE.format(ctx=clean(r["context"]), q=clean(r["question"])),
            "answerable": False,
            "category": "missing-context",
            "domain": clean(r["title"]),
            "context": clean(r["context"]),
            "scoring": {"method": "always-abstain"},
            "source": "SQuAD2",
        })
    with OUT.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    OUT_EVAL.write_text(json.dumps(eval_items, ensure_ascii=False, indent=0) + "\n")
    print(f"wrote {len(items)} items -> {OUT} and {OUT_EVAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
