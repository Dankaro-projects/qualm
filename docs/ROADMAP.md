# Roadmap

The milestones are ordered so that the cheapest test that could stop the project runs first. M1 is a stop point, not a warm up.

Each milestone states the result that would count as failure, because a milestone without a failure condition is only an item on a list.

---

## M0: Scaffold and first connection (done)

- The repository, documents, schemas and roadmap exist.
- `uv sync` works for the Python side, and the Rust toolchain is installed.
- The harness reaches Ollama and completes schema constrained probes in both directions (306 of 306).
- The harness writes a run log, and the Python side reads it back.

**Done when** one probe runs from end to end and the analysis side reads the log. This milestone has no failure condition, because it is plumbing.

---

## M1: Validity screen, a stop point (passed on 17 August 2026)

**Outcome.** The metric cannot be estimated on this item format. None of the three models makes a genuine knowledge error on the 46 answerable items, so Type 2 AUROC has no wrong answers to separate from the right ones: it is undefined, not weak. The pooled figures first published (0.978, 0.901 and 0.562; the first is 0.967 under the current command line) mostly measure the separation of unanswerable questions under another name.

The claim that the coherence hypothesis was confirmed has been withdrawn. It holds for 1 of the 3 models (granite, Fisher p = 0.0017), and the type of falsity, the frequency in the training corpus and the form of the question each explain it at least as well. The full list of withdrawn claims is in [the M1 report](../results/2026-08-17-M1-validity-screen.md).

M1 therefore did not stop the project, but it changed its direction: the item format had to change before any metric could mean anything. See M2.

**Method.** Run the five screens in [METRICS.md](METRICS.md) against `granite4.1:8b`, `qwen3.5:9b` and `gemma4:12b`, using a small pilot set of about 200 items that covers answerable, unanswerable and false premise questions. Everything else waits on this.

**Done when** each model has a pass or fail on all five screens, with bootstrap confidence intervals.

**Fails if** no model shows variance or discrimination above chance. That would mean confidence is only a linguistic construction ([RESEARCH.md, section 7](RESEARCH.md)).

**If it fails**, do not continue with verbalised confidence. Switch the elicitation to self consistency or to the paired signals before and after solving, which the design already allows ([DESIGN.md](DESIGN.md)), and run M1 again. If that also fails, the honest output is a report of a null result. Such a report is still useful and publishable, because the field is divided on exactly this question.

The first data cast doubt on screen 1, because every confidence value fell between 0.73 and 0.98. That reading was mistaken, because that task had no ground truth. With real answerable and unanswerable items, the variance screen passes. The reasoning for running M1 first remains correct.

---

## M2: Item sets (partly done)

**Done so far.** `datasets/fc-hard.jsonl` holds 57 forced choice items in 4 domains, 36 of them answerable, with 6 paraphrase pairs. It held 58 items until the audit of 17 August 2026 removed a defective one and corrected three others. Each model makes 14 to 18 genuine errors on answerable items, spread over 7 to 10 items. `datasets/pilot.jsonl`, with 102 open questions, is kept but cannot measure AUROC (see M1).

The next step was converted rather than written by hand. The unanswerable strata of the hand written set held only 1 to 5 items each, and the answerable items with errors numbered about 8, so no confidence interval was narrower than ±0.15. The audit therefore recommended converting about 300 hard MMLU-Pro items and about 200 unanswerable items (at least 40 per stratum) to the probe item schema. That takes one converter script and one night of GPU time, and it also makes the results comparable with the 33 model atlas. Writing the same number of items by hand would take weeks. See [the M3b report, sections 4 and 5](../results/2026-08-17-M3b-audit-and-rerun.md). `datasets/large-v1.jsonl` is the result: 300 MMLU-Pro items and 280 unanswerable items in 7 groups of 40.

Two ideas were dropped from this milestone. A larger control model served from the cloud was dropped because it falls outside the claim and adds an uncontrolled elicitation path; if a control is needed, `gemma4:12b` at q8_0 serves, and it also forms a pair with the same model at q4. Coherent false premises that are absent from the training corpus were dropped as a line of research (see M5).

The items, not the code, are the difficult part:

- Use the AbstentionBench taxonomy: unknown answer, underspecification, false premise, subjective and stale.
- Add the coherent false premise stratum for the coherence hypothesis ([METRICS.md](METRICS.md)).
- Include a slice stratified by domain. The quantisation finding is at domain level, so a single pooled number would hide the effect the project most expects to see.
- Make ground truth checkable by a machine. If scoring needs a person or a judge model, the cost rises sharply and a new confound enters.

**Done when** there are at least 300 hard answerable items and at least 40 items in each unanswerable stratum, all scorable by machine and versioned with identifiers, whether written or converted.

**Fails if** ground truth cannot be established without a judge model.

---

## M3: The leaderboard, the first deliverable (done on 17 August 2026)

**State.** The first report on forced choice quoted the pooled AUROC as its headline and was corrected after an audit by 13 agents. The item set and the harness were fixed, and each model was run twice with a cluster bootstrap. Every estimate now prints both the number of records and the number of clusters. The corrected report is [M3b](../results/2026-08-17-M3b-audit-and-rerun.md), and the original report, with its table of corrections, is [M3](../results/2026-08-17-M3-forced-choice.md).

Type 2 AUROC on answerable items only, with cluster confidence intervals, for run 1 and run 2:

| Model | Run 1 | Run 2 | Screen |
|---|---|---|---|
| `gemma4:12b` | **0.784 [0.552, 0.963]** | **0.815 [0.617, 0.964]** | Passed |
| `qwen3.5:9b` | 0.643 [0.419, 0.800] | 0.643 [0.419, 0.800] | Failed |
| `granite4.1:8b` | 0.520 [0.365, 0.678] | 0.544 [0.382, 0.713] | Failed |

On 57 items, one model discriminated and two did not. `gemma4:12b` and `granite4.1:8b` gave different outputs from run to run (10 and 6 of 114 records), while `qwen3.5:9b` gave identical output.

The audit set the following rules for what this milestone may claim:

- AUROC is reported on answerable items only, with the pooled figure labelled beside it, never the other way round.
- Results come from two runs, because the third decimal place of a single run is noise.
- Operating points are given as thresholds that actually occur in the data, never as a percentage of coverage.
- The second quantisation level is not the contribution, because arXiv 2604.08976 already reports AUROC₂ within one model across quantisation levels. `gemma4:12b` at q8_0 is run because it costs one download and one night, and it is reported as a replication at another pair of bit widths.
- The 33 model atlas is only comparable if MMLU items are used. The atlas uses 1,500 MMLU items, while the forced choice set has 57 items written for this project. Adopting MMLU items is the M2 decision above.
- The M ratio and meta-d′ are dropped. They are undefined on open questions, need hundreds of trials per cell on forced choice, and do not correlate across quantisation levels in any case. No decision depends on them.
- Null results are published. Two of the three models are the null result.

**Measured the same day on 300 items** ([Doors 1 and 2](../results/2026-08-17-doors-1-2.md)). The verbalised AUROC of `gemma4:12b` is **0.530 [0.476, 0.582]**, which is chance, while agreement across six option orderings gives **0.737 [0.680, 0.794]**. The prediction of arXiv 2604.24070 holds, and the 57 item finding that `gemma4:12b` has a signal did not survive.

**Done when** the leaderboard shows cluster confidence intervals from two runs, on answerable items only and pooled, with AUROC from agreement beside AUROC from verbalised confidence, plus the q8_0 replication. This was completed on 17 August 2026: see the Doors report, section 1.4 (three models with two runs each), section 1.5 (the q8_0 pair, where discrimination is unchanged and abstention moves) and section 2.6 (a second model as judge).

**Fails if**, with at least 300 hard answerable items, no model's lower confidence bound on answerable items clears about 0.55 from either signal. In that case verbalised confidence on this stack is decorative, as arXiv 2604.22215 already found for the previous generation, and the honest output is a null replication.

---

## M4: The harness, the second deliverable (built on 17 August 2026)

The harness is a subcommand of the command line tool. An earlier sketch of a `serve` mode, with a feeling of knowing before solving, a judgement of learning after, and trust, retry or aggregate branches behind an HTTP endpoint, has been dropped. An integration into a workflow automation tool was also built and tested from end to end on the same day, then set aside, because it measured nothing that the harness had not already measured.

**What the gate is, as measured** on the SQuAD 2.0 passage task with `gemma4:12b` and 300 items (`results/2026-08-17-doors-1-2.md`, section 2):

| Check | Wrong answers let through (of 84) | Answers sent for review | Cost |
|---|---|---|---|
| Stated confidence | 73 | 13 | None, and it does not help |
| The answer appears word for word in the passage | 47 | 41, of which 90% are real errors | None |
| The same answer in 3 seeded samples | 37 | 76 | Three times the inference |
| Grounded and unanimous | **28** | 86 | Three times the inference |

The order is grounding first, because it costs nothing, then agreement, and never stated confidence. The remaining errors are consistent hallucinations: an unanswerable question answered the same way every time. No check that relies only on the model's own outputs can detect them.

**`qualm-harness gate`**, in `crates/qualm-harness/src/gate.rs`:

- `--question … [--passage …]` returns JSON with `answer`, `confidence`, `agreement`, `grounded`, `malformed`, `decision` and every sample.
- `--items <jsonl> [--out …]` returns a decision for each item and the table of gated against ungated results. Items carry a `context` field for the grounding check.
- The defaults are `--samples 3 --sample-temperature 0.7 --require-grounded true --accept-at 1.0`, which is the gate as measured.

**The completion criterion was met on a fresh run** (`results/gate/gemma4-12b-squad2-gate.jsonl`, 7 minutes 43 seconds for 300 items with 3 samples each):

| Gate | Accepted | Accuracy of accepted answers | Wrong answers let through (of 76) | Sent for review |
|---|---|---|---|---|
| None | 70% | 0.640 | 76 | 0 |
| Grounded | 60% | 0.732 | 48 | 32, of which 28 are real errors |
| Unanimous | 49% | 0.736 | 39 | 63, of which 37 are real errors |
| Grounded and unanimous | 45% | 0.794 | **28** | 75, of which 48 are real errors |
| The configured decision, which also sends malformed answers for review | 43% | 0.828 | **22** | 83, of which 54 are real errors |

This has the same shape as the measurement on the analysis side (28 errors and 0.793 accuracy). Ungated task accuracy was 0.730 here against 0.713 there, which is normal variation between runs for this model.

**Fails if** gating costs more accuracy than it saves on a task that someone relies on. This must be measured, not assumed. It is a real possibility if the signal is weak, which is why M1 comes first.

---

## M5: Report

The output is a benchmark paper or a technical report with one main claim: **Type 2 AUROC of q4_K_M models of the 2026 generation on Ollama, stratified by answerability**, comparing verbalised confidence with self consistency, with cluster confidence intervals and the variation between runs, plus a replication within one model at q4 and q8. The measurement is not new in kind: arXiv 2604.22215 covers the previous generation.

A second claim, that confidence tracks coherence rather than correctness, was cut as a claim about mechanism on 17 August 2026. It rested on 1 of 3 models and 5 items with three named confounds, and arXiv 2607.08456 had already published the effect across five models. The `false-premise` and `coherent-false-premise` strata remain as ordinary strata in the leaderboard.

The report will include the harness as an artefact, and null results where those are the results.

---

## What the project needs

**Needed later**

- `gemma4:12b` at q8_0 for the replication within one model. It needs one download and must fit in 16 GB with 4 slots; `ollama ps` should report 100% GPU.
- A task with checkable outputs that people actually run, to test whether the gate saves more than it costs in practice.

**Not needed, although it may seem so**

- Fine tuning. The harness result requires no change to model parameters, and fine tuning for reasoning has been measured to reduce abstention.
- More GPU capacity. All of the work is inference with at most 4 concurrent requests.
- More models for now. Three models that produce clean JSON are enough to settle M1.

## Why M1 comes before M2

Running the screen before the real item set exists may look like the wrong order. It is deliberate. The pilot set for M1 has about 200 items and can be rough. Building 1,000 careful items before knowing whether any model has a readable signal would risk spending the most expensive effort on a hypothesis that may already be dead.
