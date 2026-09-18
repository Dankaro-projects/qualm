# Audit of M3 and the corrected rerun, 2026-08-17

**This report covers two pieces of work, because the second exists only because of the first.**

1. A 13 agent adversarial audit of everything in the repository at the time. It used six finders, each with a distinct focus (Rust scoring, statistics, items, claims, reproducibility, and schema and provenance). Each finder was followed by an independent sceptic instructed to *refute* every finding by reproducing it. One further reviewer assessed the direction of the project and fetched the abstracts of prior work. **The audit produced 24 findings, refuted 0, and the sceptics added 4.** Every finding that a conclusion depends on was then verified again by hand from the raw logs before anything below was written.
2. The corrected item set and harness were **run again, twice per model**, and scored with the corrected analysis. Those are the numbers that now stand.

The original M3 report is kept, with its corrections recorded. Its numbers reproduce exactly from its logs, which are now committed. What was wrong was the interpretation of what those numbers measure.

---

## 1. What the audit found

### Blockers: a published number or claim was wrong

| # | Finding | Where | How it was verified |
|---|---|---|---|
| 1 | **The M3 headline AUROC is the pooled construct.** Its negative class consisted of 44 to 61% *attempted unanswerable* records (granite 25/41, qwen 16/30, gemma4 11/25), which are wrong by construction. Since the M1 correction, `METRICS.md` had stated that this pooled figure "is screen 5 renamed". On answerable items only, the values are gemma4 0.849, qwen 0.643 and **granite 0.566**, so two of three models fail the discrimination screen. | M3 headline section; README; working notes; ROADMAP | recomputed from the logs; three independent scripts agree |
| 2 | **`fc-fp-king-usa` offered the true rejection of the premise ("there is no such office") as an option.** All three models chose it at confidence 1.0 and were scored as errors. It was the largest single contributor to the pooled negative class. | `scripts/make-fc.py` | mapping from choice to text through the recorded permutation |
| 3 | **The coverage and risk table was an artefact of file order.** `coverage_risk` cut the records, sorted by confidence, at 20, 40, 60 and 80% with a stable sort. 55 of gemma4's 83 attempted records were tied at 1.0, so the row "20% coverage gives 0.941" ranged from **0.765 to 1.000** when the record order was shuffled, and no confidence threshold could reach 20% coverage at all. The "actionable gate … 94% against a 69% baseline" could not be achieved by any threshold. | `metrics.py::coverage_risk`; M3 coverage table | 300 shuffles of the record order |
| 4 | **The three logs behind every M3 number were ignored by git and never added with force**, contrary to the rule in `.gitignore` itself. A fresh clone could not run the reproduce command in the report (`FileNotFoundError`), and the GitHub remote lacked the logs too. The withdrawn pilot logs *had* been committed. | `.gitignore:22`; git | a fresh clone |
| 5 | granite's `premise_challenged` rate on underspecified items was **20%** (2/10), not 40%. The 40% was the value from the run before the upgrade, left in place: **the table mixed two runs**. | M3 premise table | recount; the terminal output from before the upgrade was found in the session transcript |
| 6 | The M3 claim of 25 to 41 genuine errors per model was wrong. The genuine errors on answerable items were 16/14/14 records on **10 to 11 distinct items**, and only 4 to 5 of those items were wrong under *both* orderings. **For each model, 6 items changed correctness with the option order at identical confidence.** Roughly half of the clean negative class was noise from option position, which no confidence signal can anticipate. | M3 section on why the format changed | pairing each item across permutations |

### Major findings: a code path produced wrong numbers, or a stated guarantee did not exist

| # | Finding | Where |
|---|---|---|
| 7 | **`_twin_pairs` mixed option orderings.** It keyed originals by item id, so with two permutations the original record written *last* was paired with *both* twin records, and 6 of 12 pairs per model compared different orderings. The printed values for the paraphrase screen were wrong (gemma4 0.079, true value 0.113; qwen 0.000, true value 0.037). The verdicts did not change on this data. | `cli.py::_twin_pairs` |
| 8 | **Scoring ignored `premise_challenged`.** The prompt tells the model to choose a letter *and* flag a false premise. A model that did exactly that on an unanswerable item was scored as a confident **error** (qwen: 4 records at 0.95 to 1.0; granite: 2). The report had described the field as making the rejection of a premise "representable". | `item.rs::score_choice`; `run.rs` |
| 9 | **`fc-stale-tallest-building`**: three of the four options were completed buildings, so Jeddah Tower was the only defensible answer, and the item tested nothing about staleness. Four confident "errors" (granite ×2, gemma4 ×2) came from it. | `make-fc.py` |
| 10 | **The statement "Provenance is enforced in the schema, not by discipline" was false.** `sampling`, `harness_version`, `prompt_set` and `backend_version` were all optional; `quantisation: ""` passed validation; and nothing in Rust or Python ever loaded the record schema. A copy of the headline log with its provenance removed passed validation and still scored 0.890. | `run-record.schema.json`; working notes; DESIGN.md; README |
| 11 | **The "accidental stability check" (0.645 / 0.650 / 0.829 before the upgrade) had no surviving log.** The same `--out` path and `File::create` had overwritten it. The numbers were real terminal output, found in the session transcript, but could not be verified from the repository. The transcript also shows that **gemma4 FAILED the paraphrase screen before the upgrade (4/5)**, which the wording "barely moved … all three pass 5/5" left out. `backend_version` (Ollama 0.32.9, unchanged across the upgrade) cannot attribute a record to a state of the driver and kernel, as the paragraph claimed. | M3 stability section |
| 12 | The M3 report claimed that trailing nouns ("Sahara Desert") now score correctly. **No such logic exists.** `norm()` strips leading articles only, so "Sahara Desert" and "Naypyitaw" still score as wrong. Only the fix for numbers written as words was real. | M3 section on what changed |
| 13 | The working notes said that the problem of confidence with two meanings "needs prompt set v2". However, the v2 template is **byte for byte identical** to v1 on that instruction; v2 only appends the suffix for the choice. M3 then reported again the exact contrast in mean confidence (granite obvious FP "0.500") that M1 §4 had withdrawn as an artefact of the abstention convention: it consists of four attempted rows at about 1.0 and four abstained rows at 0.0. | Working notes, known problems; M3 coherence table |

### Minor findings that were also fixed

- `fc-unk-ants` said "exactly" while its options said "about", and one option was within 25% of the published estimate.
- `fc-largest-cell` asked for the "largest" cell without naming a measure, which admits "a motor neuron" (the longest) and "a muscle fibre" (the largest by volume).
- M3 said that all models missed `fc-human-bones-baby` and `fc-hundred-years-war`. The logs show 2/6 and **3/6** correct, and the second was answered with *position A in all six records*.
- "granite states 0.991 while scoring 0.577" paired the confidence on the answerable stratum with the accuracy on all attempted items. Compared like for like, the figures are 0.961 against 0.577, or 0.991 against 0.778.
- The working notes quoted qwen's M1 AUROC as 0.978. The current analysis on the committed log gives 0.967, because the old analysis leaked one abstention into the negative class.
- The statement that the unanswerable strata held 4 to 5 items each was out of date: the stale and subjective strata had 2.
- The baseline report cited "36 samples", while its own table sums to 40.
- `probe-response.schema.json` said that `choice` is null when the model abstains. The forced choice format for each item forbids null, so all 80 abstained forced choice records carry a letter forced by the grammar.
- Sections in four files were out of date (Rust "not installed", "no dataset yet", and `datasets/README.md` described as "deliberately empty").

### Cleared: checked and found correct

This is worth stating, because it limits the damage: **the core of the harness is right.**

- For all 348 records, `correct` matches the comparison between the chosen letter under the recorded permutation and the answer text, with 0 mismatches.
- The enum for each item matches the letters shown in the prompt (captured with wiremock).
- `validate()` accepts exactly the documented set over 1024 enumerated inputs.
- The request carries temperature 0, seed 0, `num_ctx`, `num_predict` and `think:false`, and `total_ms` includes time waiting in the queue.
- The AUROC values 0.890/0.656/0.650, the ECE, the error counts, the abstention rates and the stratum tables all reproduce in three independent recomputations.
- The dataset regenerates byte for byte from its script, and a fresh clone builds, lints, tests and validates exactly as CI does.
- All six logs validate against the record schema.
- The ground truth of every forced choice item was checked and is correct (Everest 8849 m, 116 years, 2 August 1776, about 300 bones, and so on).

---

## 2. What was changed

**Harness 0.3.0.** `Item::with_premise_challenge`: a rejection of the premise on an unanswerable item is a correct refusal (`correct: null`, like a correct abstention); on an answerable item it is an error, whatever the letter. No answerable record in any run carried the flag, so the second rule is a decision rather than a fit to the data.

**Items** (`fc-hard.jsonl`, now **57**). The distractor in `fc-fp-king-usa` was replaced ("the Chief Justice"). `fc-stale-tallest-building` was removed, which leaves the stale stratum with 1 item, too few to report. The options of `fc-unk-ants` were made exact integers to match "exactly". `fc-largest-cell` now says "by diameter".

**Analysis.**

- A cluster bootstrap (`bootstrap(..., clusters=)`), in which each cluster is an item and each twin is folded into its original. Every `Estimate` prints both the record count n *and* the number of clusters.
- `_twin_pairs` now keys on the pair of item and permutation.
- AUROC is reported **for answerable items only and pooled, both labelled, with the negative class broken down**.
- `coverage_risk` was replaced by `operating_points`, which gives the coverage and retained accuracy at every threshold that *actually exists*.
- "Declined" now means abstained or premise rejected.
- The separation gap on attempted items only is printed beside screen 5 as a caution.

**Schema.** `sampling` (with `temperature/seed/num_ctx/num_predict/num_parallel`), `harness_version` and `prompt_set` are now **required**, and `quantisation` must not be empty. All committed logs still validate, and a record with its provenance removed now fails. `probe-response.schema.json` describes what is actually sent.

**Archive.** The 0.2.0 forced choice logs and both 0.3.0 runs are committed with `git add -f`, as is `results/raw/quarantine/README.md`.

---

## 3. The rerun

**57 items × 2 orderings × 3 models × 2 runs = 684 probes, 0 failures.** Harness **0.3.0**, prompt set `v2`, Ollama 0.32.9, all models **q4_K_M**, temperature 0, seed 0, `num_ctx` 8192, `num_predict` 512, `think:false`, 4 concurrent requests. The inference machine ran kernel 7.0.0-28 and driver 580.178.04. The two runs of each model were made one after the other, between about 13:27 and 13:32 UTC.

```bash
cd python
uv run python -m qualm_analysis.cli \
  ../results/raw/granite4-1-8b-fc-0.3.0-r1.jsonl \
  ../results/raw/qwen3-5-9b-fc-0.3.0-r1.jsonl \
  ../results/raw/gemma4-12b-fc-0.3.0-r1.jsonl
# and the same with -r2
```

### The headline, on the clean construct

Type-2 AUROC on **answerable items only**, with a cluster bootstrap over 32 items (36 answerable items, with 4 twins folded in) and a 95% percentile interval:

| Model | run 1 | run 2 | errors (records / items) | screen |
|---|---|---|---|---|
| `gemma4:12b` | **0.784 [0.552, 0.963]** | **0.815 [0.617, 0.964]** | 15 / 7 | 5/5 PASS |
| `qwen3.5:9b` | 0.643 [0.419, 0.800] | 0.643 [0.419, 0.800] | 14 / 8 | **FAIL** discrimination |
| `granite4.1:8b` | 0.520 [0.365, 0.678] | 0.544 [0.382, 0.713] | 18 / 10 · 17 / 9 | **FAIL** discrimination |

For comparison, the **pooled** figure includes attempted unanswerable records in the negative class. Do not quote it without saying so:

| Model | run 1 | run 2 | negative class |
|---|---|---|---|
| `gemma4:12b` | 0.825 [0.656, 0.951] | 0.829 [0.685, 0.940] | 22 to 23 = 15 answerable + 7 to 8 unanswerable |
| `qwen3.5:9b` | 0.641 [0.482, 0.777] | 0.641 | 23 = 14 + 9 |
| `granite4.1:8b` | 0.650 [0.506, 0.785] | 0.668 [0.522, 0.805] | 38 to 39 = 17 to 18 + 21 |

**One model has a confidence signal that discriminates on this set; two do not.** gemma4's lower bound is above 0.5 in both runs, but only by 0.05 and 0.12; with 7 items in error, that is the margin. gemma4 also sits *at* the threshold of the paraphrase stability screen (mean |Δ| 0.146 / 0.150 against a gate of 0.15). One more sensitive twin and it would fail screen 3.

### The numbers changed between runs, and the size of the change matters

The items and settings were the same, and the runs were minutes apart:

| Model | records that differ between run 1 and run 2 (of 114) | AUROC on answerable items, run 1 and run 2 |
|---|---|---|
| `qwen3.5:9b` | **0** | 0.643 and 0.643 |
| `granite4.1:8b` | 6 | 0.520 and 0.544 |
| `gemma4:12b` | 10 | 0.784 and 0.815 |

Temperature 0 with a fixed seed is **not deterministic** for two of the three models with 4 concurrent slots. The inference is batched, and the kernel scheduling depends on what else is in the batch. qwen's output is identical bit for bit. Consequently, **the third decimal place of a single run is noise for gemma4 and granite**, the M3 "stability check" before and after the upgrade was comparing values within this noise, and every number in this report is quoted from both runs for that reason. Whether the nondeterminism disappears with one concurrent request was not tested.

### Operating points that a gate can actually use (answerable items)

| Model | threshold | coverage | retained accuracy | baseline |
|---|---|---|---|---|
| `gemma4:12b` | conf ≥ 1.0 | 73 to 75% | **0.907 / 0.923** | 0.79 |
| `gemma4:12b` | conf ≥ 0.95 | 87 to 88% | 0.871 / 0.873 | |
| `qwen3.5:9b` | conf ≥ 1.0 | 44% | 0.906 | 0.81 |
| `granite4.1:8b` | conf ≥ 1.0 | 57 to 58% | 0.780 / 0.810 | 0.75 to 0.76 |

This is all that M3 offers the harness that can be used: **gemma4 at a stated confidence of 1.0 keeps about 3/4 of answerable items and is right on about 91 to 92% of them, against 79% without the gate.** This rests on 36 items, so it should be treated as a hypothesis with a wide interval, not as a specification. granite's gate gains almost nothing.

### Declining, and the rejection of premises

"Declined" means abstained or premise rejected. The recall on the 42 unanswerable records is 0.81 to 0.83 for gemma4, 0.79 for qwen and 0.50 for granite. Unnecessary declining on answerable items is 0 for every model, apart from one gemma4 record in run 2. The rejection of premises by category (run 1 / run 2):

| Model | obvious FP | coherent FP | underspecified | unknown-answer |
|---|---|---|---|---|
| `granite4.1:8b` | 75% / 75% | 0% / 0% | 10% / 30% | 75% / 75% |
| `gemma4:12b` | 75% / 75% | 40% / 30% | 20% / 10% | n/a |
| `qwen3.5:9b` | 50% / 50% | 30% / 30% | n/a | n/a |

Each cell rests on 4 to 5 items. Read the table for direction, not magnitude.

### The coherence rows, with the caution they always needed

The mean confidence over **all** rows, for answerable items, coherent false premises and obvious false premises, is 0.992 / 0.990 / 0.500 for granite, 0.972 / 0.955 / 0.956 for qwen, and 0.98 / 0.32 / 0.27 for gemma4. granite's coherent false premise row still cannot be told apart from its answerable row, and granite still declines 0% of coherent false premises against 75% of obvious ones. **However**, granite's obvious false premise value of 0.500 consists of four attempted rows at about 1.0 and four abstained rows at 0.0, which is the abstention convention artefact that M1 §4 withdrew. The *contrast* in confidence is therefore not the evidence; the decline rate of 0% against 75% is. The finding still rests on 1 of 3 models and 5 items, and the type of falsehood and the frequency in the corpus remain unaddressed. arXiv 2607.08456 has since shown, across five models of 2 to 14B parameters, that confidence in the answer is "nearly blind" to answerability on natural false premise data. That is the granite pattern, generalised and published. This line of work is therefore **dropped as a claim about mechanism** and kept only as an ordinary stratum.

---

## 4. The position of the project after the audit

**Established:** on 57 authored forced choice items, one of three q4_K_M models from 2026 has verbalised confidence that discriminates its own errors better than chance (lower bound 0.55 to 0.62), and its only usable gate is "stated confidence == 1.0". The other two models are at or near chance. Each model's confidence collapses onto 3 to 8 distinct values. Two of the three models are not deterministic from run to run at temperature 0.

**Not established:** anything about quantisation (only one level was run), effects by domain (58 items across four informal domains cannot support any), the coherence mechanism, or comparability with any published atlas.

**Prior work that the contribution section undercounted.** arXiv 2604.22215 ran seven open weight models of 3 to 9B parameters at Q5_K_M GGUF on a 16 GB consumer GPU, with verbalised confidence, a validity screen and AUROC₂. All seven failed the screen (a ceiling rate of 91.7%). It has no unanswerable items. arXiv 2607.08456 covers the dimension of false premises and answerability. What remains new here is narrow: models from the 2026 generation, q4 rather than q5, Ollama, and forced choice **stratified by answerability** with a field for rejecting the premise. The comparison of quantisations within one model, which the roadmap called "the contribution", replicates 2604.08976 at another pair of bit widths. It is worth doing because it is cheap, not because it is new.

**The direction reviewer's verdict, which the numbers support:** the constraint is the sample size *n*, not the code. Every cycle of withdrawal in this project would have been unnecessary with 300 items. The recommendation is to stop writing items by hand, and instead to reuse MMLU-Pro (hard answerable items, about 300) and AbstentionBench (unanswerable items, about 200, at least 40 per stratum), converted to the probe item schema. Running 6 permutations would allow `signals.agreement` (consistency across orderings) to be measured against verbalised confidence. 2604.24070 predicts that agreement will dominate. That comparison is both the cheapest result worth citing and the number a production gate actually needs. It amounts to about 21k probes, one night of GPU time.

## 5. Next steps

1. **Decide the strategy for items**: write them or reuse them. This changes what the project is, so it is not decided here.
2. If the choice is to reuse: build a converter from MMLU-Pro and AbstentionBench to `probe-item.schema.json`, using the existing validator, with 6 permutations and two runs.
3. Populate `signals.agreement`, and report the AUROC of agreement against the AUROC of verbalised confidence.
4. Check determinism with one concurrent request for gemma4 and granite. This is cheap and would settle whether the noise comes from batching.
5. Remove from the roadmap: the estimation of M-ratio and meta-d′, the cloud "control model", the Rust `serve` mode until a flow with checkable output exists, the line of work on the coherence mechanism, and the target of "≥1000 authored items".
