# Quarantined logs

The files in this folder are kept as evidence, **not** as data. They must never be passed to the analysis.

## `granite-pilot-STALE.jsonl`

This file holds a second `granite4.1:8b` run over the same 102 items, with the same harness version and the same `temperature: 0, seed: 0`. It is quarantined for two reasons.

1. **All 102 records violate `run-record.schema.json`**, because they have no `item` block. A binary built before that field existed wrote them. `cargo clippy` and `cargo test` do not refresh `target/debug/<bin>`, so the stale binary was used without anyone noticing. `cli.py` crashes on this file with `KeyError: 'item'`.
2. The records carry `harness_version: "0.1.0"`, which is **identical to the valid logs**, because the version was never raised when the record shape changed. The schema describes this field as identifying the code version, and this file shows that it does not.

**The reason it is kept:** it is the evidence that this harness is **not deterministic at temperature 0 and seed 0**. It disagrees with `../granite4-1-8b-pilot.jsonl` on 5 to 7 items, and three of them fall inside the 13 item stratum of coherent false premises that carries the M1 headline. On this file, that stratum gives 23% abstention rather than 31%. Deleting the file would delete the evidence against a claim of reproducibility.
