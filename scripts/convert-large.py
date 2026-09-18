#!/usr/bin/env python3
"""Build datasets/large-v1.jsonl from public sources — the Door 1 set.

Why converted rather than authored: hand-authoring reached 57 items with strata
of 1–5, and every CI computed on that was ±0.15 or wider. The 2026-08-17 audit's
direction review put it plainly: the constraint is n, not code. These sources
give ≥40 per stratum in an afternoon and buy comparability with the atlas that
uses MMLU items.

Sources (all fetched over HTTPS, cached under data-cache/, sampled with a fixed
seed so the file regenerates byte-identically):

  answerable      MMLU-Pro test split (TIGER-Lab/MMLU-Pro, MIT). 10-option
                  forced choice, stratified across its 14 categories. Hard for
                  8–12B models by construction — that is what buys a negative
                  class without tuning difficulty by hand.
  false-premise   FalseQA test (thunlp/FalseQA, label==1) + KUQ 'false assumption'
  unknown-answer  KUQ 'future unknown' + 'unsolved problem'
  subjective      KUQ 'controversial'
  underspecified  KUQ 'ambiguous'
  counterfactual  KUQ 'counterfactual'
                  (KUQ = amayuelas/KUQ, knowns_unknowns.jsonl)
  stale           NOT AVAILABLE from these sources — stated, not papered over.

Unanswerable items are OPEN-ENDED (no options; scoring always-abstain), because
their sources carry no distractors and inventing them is the hand-authoring we
are stopping. This is a FORMAT CONFOUND against the forced-choice answerable
items: any answerable-vs-unanswerable contrast (screen 5, decline rates) is
also a format contrast. AUROC on the answerable-only construct is unaffected.
"""

from __future__ import annotations

import ast
import io
import json
import random
import sys
from pathlib import Path

import pandas as pd
import requests
import datasets

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data-cache"
OUT = ROOT / "datasets" / "large-v1.jsonl"
SEED = 20260817
N_ANSWERABLE = 300
N_PER_UNANSWERABLE_SOURCE = 40
MAX_PROMPT_CHARS = 1500  # keeps every prompt well inside num_ctx 8192 at 10 options


def clean(s: str) -> str:
    return " ".join(str(s).split())


def mmlu_pro(rng: random.Random) -> list[dict]:
    ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=str(CACHE))["test"]
    by_cat: dict[str, list] = {}
    for r in ds:
        opts = r["options"]
        if isinstance(opts, str):
            opts = ast.literal_eval(opts)
        # Ten options exactly, all distinct, answer index in range, prompt short
        # enough. Anything else is skipped rather than patched.
        if len(opts) != 10 or len(set(map(clean, opts))) != 10:
            continue
        ai = int(r["answer_index"])
        if not 0 <= ai < 10:
            continue
        if len(r["question"]) > MAX_PROMPT_CHARS or any(len(o) > 300 for o in opts):
            continue
        by_cat.setdefault(r["category"], []).append((r, [clean(o) for o in opts], ai))
    cats = sorted(by_cat)
    per = N_ANSWERABLE // len(cats)
    extra = N_ANSWERABLE - per * len(cats)
    items = []
    for i, c in enumerate(cats):
        pool = sorted(by_cat[c], key=lambda t: int(t[0]["question_id"]))
        rng.shuffle(pool)
        take = per + (1 if i < extra else 0)
        for r, opts, ai in pool[:take]:
            items.append({
                "id": f"mp-{r['question_id']}",
                "prompt": clean(r["question"]),
                "answerable": True,
                "category": "answerable",
                "domain": c,
                "options": opts,
                "scoring": {"method": "choice", "answer": opts[ai]},
                "source": "MMLU-Pro",
            })
    return items


def kuq(rng: random.Random) -> list[dict]:
    ds = datasets.load_dataset(
        "amayuelas/KUQ", data_files="knowns_unknowns.jsonl", cache_dir=str(CACHE)
    )["train"]
    cat_map = {
        "false assumption": "false-premise",
        "future unknown": "unknown-answer",
        "unsolved problem": "unknown-answer",
        "controversial": "subjective",
        "ambiguous": "underspecified",
        "counterfactual": "counterfactual",
    }
    by_src: dict[str, list] = {}
    for i, r in enumerate(ds):
        if not r["unknown"] or r["category"] not in cat_map:
            continue
        q = clean(r["question"])
        if not q.endswith("?") or len(q) > MAX_PROMPT_CHARS:
            continue
        by_src.setdefault(r["category"], []).append((i, q))
    items = []
    for src in sorted(by_src):
        pool = sorted(by_src[src])
        rng.shuffle(pool)
        for i, q in pool[:N_PER_UNANSWERABLE_SOURCE]:
            items.append({
                "id": f"kuq-{i}",
                "prompt": q,
                "answerable": False,
                "category": cat_map[src],
                "domain": "general",
                "scoring": {"method": "always-abstain"},
                "source": f"KUQ/{src.replace(' ', '-')}",
            })
    return items


def falseqa(rng: random.Random) -> list[dict]:
    p = CACHE / "falseqa-test.csv"
    if not p.exists():
        t = requests.get(
            "https://raw.githubusercontent.com/thunlp/FalseQA/refs/heads/main/dataset/test.csv",
            timeout=60,
        )
        t.raise_for_status()
        p.write_text(t.text)
    df = pd.read_csv(io.StringIO(p.read_text()))
    pool = [(i, clean(q)) for i, (q, lab) in enumerate(zip(df["question"], df["label"])) if int(lab) == 1]
    rng.shuffle(pool)
    return [{
        "id": f"falseqa-{i}",
        "prompt": q,
        "answerable": False,
        "category": "false-premise",
        "domain": "general",
        "scoring": {"method": "always-abstain"},
        "source": "FalseQA",
    } for i, q in pool[:N_PER_UNANSWERABLE_SOURCE]]


def main() -> int:
    rng = random.Random(SEED)
    items = mmlu_pro(rng) + falseqa(rng) + kuq(rng)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(items)} items -> {OUT}")
    print("by category:", dict(sorted(Counter(i['category'] for i in items).items())))
    print("by source:  ", dict(sorted(Counter(i['source'] for i in items).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
