#!/usr/bin/env -S uv run --quiet --with jsonschema python
"""Validate a probe-item JSONL set.

Schema conformance is the cheap half. The expensive mistakes in an item set are
semantic and a validator can catch several of them:

  - duplicate ids            -> results silently collide
  - answerable/category      -> disagreement makes abstention scoring wrong
  - always-abstain misuse    -> an answerable item that can never be right
  - dangling paraphrase_of   -> validity screen #3 gets fewer twins than assumed
  - thin strata              -> a category with 3 items yields a meaningless rate

Usage:  scripts/validate-items.py datasets/pilot-200.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

REPO = Path(__file__).resolve().parent.parent
MIN_PER_STRATUM = 20  # below this a per-category rate is noise, not a measurement

# Which categories must be answerable, per schemas/probe-item.schema.json.
ANSWERABLE_CATEGORIES = {"answerable"}  # every other enum value must be answerable=false


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1

    schema = json.loads((REPO / "schemas" / "probe-item.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    items: list[dict] = []
    errors: list[str] = []

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: not valid JSON - {exc}")
            continue
        for err in validator.iter_errors(item):
            loc = "/".join(str(p) for p in err.path) or "(root)"
            errors.append(f"line {lineno}: {loc}: {err.message}")
        items.append(item)

    # ── semantic checks ─────────────────────────────────────────────────────
    ids = [i.get("id") for i in items if "id" in i]
    for dup, count in Counter(ids).items():
        if count > 1:
            errors.append(f"duplicate id {dup!r} appears {count} times")

    id_set = set(ids)
    for item in items:
        iid = item.get("id", "?")
        cat = item.get("category")
        answerable = item.get("answerable")
        method = item.get("scoring", {}).get("method")

        if cat in ANSWERABLE_CATEGORIES and answerable is not True:
            errors.append(f"{iid}: category {cat!r} implies answerable=true")
        if cat not in ANSWERABLE_CATEGORIES and answerable is not False:
            errors.append(f"{iid}: category {cat!r} implies answerable=false")

        if method == "always-abstain" and answerable:
            errors.append(f"{iid}: always-abstain on an answerable item can never be correct")
        if method != "always-abstain" and answerable is False:
            errors.append(
                f"{iid}: unanswerable item scored with {method!r}; use always-abstain"
            )
        if method in {"exact", "numeric", "set-match"} and "answer" not in item.get("scoring", {}):
            errors.append(f"{iid}: scoring.method {method!r} needs scoring.answer")
        if method == "regex" and "pattern" not in item.get("scoring", {}):
            errors.append(f"{iid}: scoring.method 'regex' needs scoring.pattern")

        twin = item.get("paraphrase_of")
        if twin and twin not in id_set:
            errors.append(f"{iid}: paraphrase_of {twin!r} is not an id in this set")

    # ── stratum sizes: warnings, not errors ────────────────────────────────
    warnings: list[str] = []
    by_cat = Counter(i.get("category") for i in items)
    by_domain = Counter(i.get("domain") for i in items)
    for cat, n in sorted(by_cat.items()):
        if n < MIN_PER_STRATUM:
            warnings.append(f"category {cat!r} has only {n} items (<{MIN_PER_STRATUM})")
    if not any(i.get("paraphrase_of") for i in items):
        warnings.append("no paraphrase twins: validity screen #3 cannot run")
    if len(by_domain) < 2:
        warnings.append(
            "fewer than 2 domains: the quantisation effect is reported at domain "
            "level, so a single-domain set cannot show it"
        )

    # ── report ─────────────────────────────────────────────────────────────
    print(f"{path}: {len(items)} items")
    print(f"  categories: {dict(sorted(by_cat.items()))}")
    print(f"  domains:    {dict(sorted(by_domain.items()))}")
    twins = sum(1 for i in items if i.get("paraphrase_of"))
    print(f"  paraphrase twins: {twins}")

    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} error(s)", file=sys.stderr)
        return 1
    print("\nOK" + (f" ({len(warnings)} warning(s))" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
