# Design

## The division between Rust and Python

The division is drawn for reasons of maintenance and correctness, not performance. At this project's scale, thousands of probes rather than millions, either language would be fast enough for both halves.

| | Rust: `crates/qualm-harness` | Python: `python/qualm_analysis` |
|---|---|---|
| Responsible for | The control loop, concurrency, transport, schema enforcement and provenance | Type 2 signal detection, AUROC, validity screening, plots and tables |
| Runs | In production, on every request | Offline, over recorded runs |
| Changes | Rarely, and every change is a risk to correctness | Often, because this is where the analysis is explored |
| A failure means | A stuck workflow or a wrong gate decision | A wrong number in a draft |

The two halves are kept apart because they change at opposite rates. The harness must be stable because it gates real work. The analysis must be easy to change, because that is the nature of research. If the two were merged, every exploratory change to the statistics would touch code that runs in production.

**Rust is used for the loop** because the loop must keep a number of probes in flight against Ollama's 4 parallel slots without oversubscribing them, retry deterministically, enforce the response schema at the boundary, and never silently accept a malformed field. Static types at the JSON boundary are the main reason. The most common bug in this field is a field that was quietly missing.

**Python is used for the statistics** because the tools of the metacognition literature exist in Python and R. Estimation of meta-d′ (HMeta-d is a hierarchical Bayesian model), AUROC, bootstrap confidence intervals and the plots all have mature implementations. Writing Type 2 signal detection again in Rust would produce a new, unvalidated implementation of subtle mathematics, which is exactly what should be avoided when the output is a published claim.

### The interface between the two halves

The only thing that passes between the halves is a run log in JSONL. Rust writes it and Python reads it.

There is no foreign function interface, no PyO3 binding and no shared memory. The boundary is a file on disk because that file is also the archive: a reviewer can score it again, and a result can be recomputed a year later when both halves have changed. A run log remains valid input indefinitely; a linked binary does not.

```
                probe items                    run log (JSONL)
datasets/*.jsonl ──────────► qualm-harness ──────────────────► qualm_analysis ──► results/
                              (Rust)          append only         (Python)         tables, plots
                                 │                                                 leaderboard
                                 └──► Ollama HTTP API
```

## The control loop

The loop follows [RESEARCH.md, section 4](RESEARCH.md) (the Nelson and Narens model; arXiv 2605.14186). Two elicited signals surround each attempt:

```
  ┌─ FOK ─┐          ┌─ solve ─┐          ┌─ JOL ─┐         ┌── control ──┐
  │ pre   │ ───────► │ attempt │ ───────► │ post  │ ──────► │ trust       │
  │ solve │          │         │          │ solve │         │ retry       │
  └───────┘          └─────────┘          └───────┘         │ aggregate   │
                                                            │ ABSTAIN     │
                                                            └─────────────┘
```

The `ABSTAIN` branch is this project's addition and is not in the source paper. The paper optimises accuracy. This project also cares about declining to answer, which is a fourth branch and the most important one for professional document work.

The harness has two subcommands that share the client, the prompt and the response schema:

- `bench` runs every probe once, records everything and acts on nothing. It produces the run log.
- `gate` answers one question, or a batch with ground truth, and routes each answer to accept, review or decline based on grounding and agreement. It produces gated output.

They share one code path deliberately. A benchmark that does not exercise the production path measures something other than production, and that is the failure this repository is about.

## Elicitation is designed to be replaceable

Verbalised confidence is the default, not an assumption. It is also the weakest part of the design: it is overconfident throughout, clustering between 80% and 100% ([RESEARCH.md, section 5](RESEARCH.md)), and this project's first data reproduced that pattern exactly.

The elicitation strategy is therefore an interface from the start:

| Strategy | Status | Note |
|---|---|---|
| Verbalised, through the `confidence` field in the schema | Default | Parses reliably on granite, qwen and gemma |
| Self consistency, sampling several times and measuring agreement | In use in the `gate` subcommand and the agreement measurements | Needs no log probabilities, but costs one inference per sample |
| Derived from log probabilities | Blocked | Ollama does not expose token log probabilities reliably |
| A feeling of knowing before solving and a judgement of learning after | Planned | The source paper's own method |

If the validity screen rejects verbalised confidence, the project changes the strategy rather than stopping. Making the strategy replaceable is what turns a failed screen into a finding instead of a dead end.

## Provenance is part of the structure

Every row of the run log records the model, the quantisation tag, `num_ctx`, `num_parallel`, the sampling parameters, the prompt set identifier, the harness version and a timestamp. The quantisation tag is not optional metadata. As [RESEARCH.md, section 2](RESEARCH.md) explains, calibration cannot be compared across quantisation levels, so a row without a quantisation tag cannot be placed on a leaderboard at all.

From harness version 0.3.0, `schemas/run-record.schema.json` requires these fields, and CI checks every committed log against the schema. The harness does not enforce the schema at the moment of writing: it writes the fields by convention and does not load the schema. Before version 0.3.0, `sampling`, `harness_version` and `prompt_set` were optional, so a log without provenance still validated and was scored. A field is only enforced once it is listed as required.

## Where each part runs

- The harness can run on any machine that can reach an Ollama server. It uses `http://localhost:11434` by default; set `QUALM_OLLAMA` or `--ollama` for another address.
- The measurements in `results/` used a server with 4 parallel slots. Concurrency above that number only queues requests inside the server, and on the larger models a context above about 16,000 tokens moves part of the weights from the GPU to the CPU.
- The analysis runs anywhere, because it only reads run logs.

## What this project does not do

- **It does not fine tune models.** The harness result requires no change to model parameters, and fine tuning for reasoning has been measured to reduce a model's willingness to abstain. Any fine tuning would be a separate milestone with its own justification.
- **It is not a general evaluation framework.** Many exist already. This project measures one thing.
- **It is not a serving layer.** Ollama serves the models; the harness sits in front of it.
