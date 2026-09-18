# The change to forced choice and the first measurement that can be estimated, 2026-08-17

**This report has been superseded by [2026-08-17-M3b-audit-and-rerun.md](2026-08-17-M3b-audit-and-rerun.md).** It was corrected on 2026-08-17 after a second adversarial audit. Every number below reproduces from its logs (`results/raw/*-fc.jsonl`). Those logs were not committed at first, which was itself one of the audit's findings. What was wrong was the interpretation of what the numbers measure, and several statements made about them. The "Corrections" section at the end lists each withdrawn statement. The M3b report carries the full list of findings, the fixes, and a rerun of two runs on the corrected item set.

**58 items × 2 option orderings × 3 models = 348 probes, 0 failures, 348/348 valid against the schema.** Items: `datasets/fc-hard.jsonl`. Prompt set `v2`, harness **0.2.0**, Ollama **0.32.9**, all models **q4_K_M**, temperature 0, seed 0, `num_ctx` 8192, `think:false`, 4 concurrent requests, `--permutations 2`.

**Hardware and software:** an RTX 5060 Ti 16 GB, driver **580.178.04**, kernel **7.0.0-28**.

**A stability check that happened by accident.** The inference machine was upgraded during the session (kernel 6.17.0-35 to 7.0.0-28, driver 580.167.08 to 580.178.04), and every model was run again afterwards. The numbers moved very little: granite from 0.645 to **0.656**, qwen from 0.650 to **0.650**, and gemma4 from 0.829 to **0.890**. Two models are stable within noise. gemma4 moved most, which is consistent with it having the steepest distribution of confidence. This was not a controlled experiment, because the driver and the kernel changed together, but it was reassuring. It is also the reason why `backend_version` is now recorded on every record: before this, no log could be attributed to a state of the software at all.

```bash
cd python && uv run python -m qualm_analysis.cli \
  ../results/raw/granite4-1-8b-fc.jsonl \
  ../results/raw/qwen3-5-9b-fc.jsonl \
  ../results/raw/gemma4-12b-fc.jsonl
```

## Why the format changed

The open pilot could not measure Type-2 AUROC. All three models answered **all 46** answerable items correctly, so the negative class was empty and the metric was undefined. Separately, meta-d′ and M-ratio require a Type-1 decision between two alternatives, which open questions do not provide (arXiv 2603.25112).

Forced choice fixes both problems. It worked: the item set produces **25 to 41 errors per model**, so there is at last something to discriminate.

## The headline

| Model | Type-2 AUROC | negative class | accuracy | ECE | screens |
|---|---|---|---|---|---|
| `gemma4:12b` | **0.890** [0.798, 0.968] | 25 errors | 0.699 | 0.214 | **5/5 PASS** |
| `granite4.1:8b` | 0.656 [0.549, 0.754] | 41 errors | 0.577 | 0.384 | **5/5 PASS** |
| `qwen3.5:9b` | 0.650 [0.548, 0.750] | 30 errors | 0.659 | 0.309 | **5/5 PASS** |

**All three models pass all five validity screens.** This is the first time any model has done so, and the first time `order-stability` has run at all, because it needs permutations that did not exist before.

### `gemma4:12b` was described wrongly twice

On the open pilot it was called the **negative control** (an AUROC at chance, three distinct confidence values, and 1.00 on every unanswerable category). On forced choice it has the **best discrimination of the three, 0.890 [0.798, 0.968]**, as well as the best calibration (ECE 0.214) and the highest abstention recall (0.750).

The earlier reading was an artefact of the format, not a property of the model. On open items its confidence was fixed at 1.0 because it had nothing to be uncertain about. A forced choice among plausible near misses gives it a real decision, and its confidence then tracks correctness better than that of either alternative model. Its coverage curve is the most usable of the three:

| coverage | retained accuracy |
|---|---|
| 20% | **0.941** |
| 40% | 0.970 |
| 60% | 0.920 |
| 80% | 0.848 |
| 100% | 0.699 |

That is a gate the model could act on: if it declines the bottom 80% by confidence, accuracy on the remainder is 94%, against a baseline of 69%. The curve for `granite4.1:8b` is monotone but shallow (0.842 to 0.577), which is consistent with its lower AUROC.

**Calibration is a separate and worse problem.** granite states a mean confidence of **0.991** while scoring **0.577**, which gives an ECE of 0.384. Its confidence orders its answers usefully (an AUROC of 0.656 is above 0.5), yet the values themselves mean almost nothing. This is exactly why AUROC and ECE are reported separately.

## The coherence prediction, now tested on the dependent variable registered in advance

`docs/METRICS.md` registered this prediction in advance in terms of **confidence**: *"on false premises that are internally consistent, confidence should be high and flat, indistinguishable from genuinely answerable items"*. It registered the test as *"confidence separates (a) answerable from (b) obviously unanswerable but not from (c) coherent false premise"*.

Mean confidence:

| Model | answerable | coherent FP | Δ | obvious FP | Δ |
|---|---|---|---|---|---|
| `granite4.1:8b` | 0.991 | **0.990** | **+0.001** | 0.500 | +0.491 |
| `qwen3.5:9b` | 0.972 | 0.955 | +0.017 | 0.969 | +0.003 |
| `gemma4:12b` | 0.981 | 0.190 | +0.791 | 0.500 | +0.481 |

**For `granite4.1:8b`, the prediction registered in advance is met precisely.** Confidence on coherent false premises (0.990) cannot be told apart from confidence on answerable items (0.991), while obvious false premises separate clearly (0.500). granite also abstains on **0%** of coherent false premise items, against 50% of obvious ones. This time the measure is the dependent variable registered in advance, not a substitute, and the direction was predicted beforehand.

**It still holds in only 1 of 3 models.** `gemma4:12b` separates coherent false premises very strongly (Δ+0.791), so it is not misled at all. `qwen3.5:9b` separates nothing, not even the obvious false premises, so its row carries no information.

**The sample is small and the records are clustered.** The coherent false premise stratum contains **5 items** (10 records, because the same 5 items appear under 2 orderings). Records from different permutations are **not independent**, yet both the bootstrap and the reported `n` treat them as independent, which overstates the precision. The same applies to paraphrase twins. **Do not quote a confidence interval for the coherence rows**, and read every `n` in this report as roughly half its printed value in effective terms. A bootstrap that accounts for clustering is required before publication.

The confounds found in the pilot also remain. The obvious false premise items still presuppose **referents that do not exist**, while the coherent items presuppose **false relations between real entities**. All the coherent items are still **famous misconceptions**, so frequency in the training corpus remains a plausible alternative explanation. The missing experiment is still a set of items absent from the corpus and matched on the type of falsehood.

## `premise_challenged`, the field that was missing

This field was added because the instrument could not represent the rejection of a premise. gemma4's best open answer (*"1921 (for the photoelectric effect)"*, which correctly rebutted a false premise) had been scored as credulity, because the only possible moves were to answer or to abstain.

The models use the field, and it reveals behaviour that was invisible before:

| Model | obvious FP | coherent FP | underspecified | unknown-answer |
|---|---|---|---|---|
| `granite4.1:8b` | 75% | 0% | 40% | 50% |
| `gemma4:12b` | 50% | **40%** | 10% | n/a |
| `qwen3.5:9b` | 38% | 30% | n/a | n/a |

gemma4 flags 40% of coherent false premises as false. The previous instrument would have recorded each of these responses as either an error or an abstention, with no way to distinguish it from credulity.

## What changed in the harness, and what the changes caught

The new validation proved its value at once. The first forced choice run recorded **50 of 60 probes as failures**, because a single shared response schema made `answer` required and `choice` optional, so every model filled `answer` and left `choice` empty. The old code would have scored these silently as 50 confident wrong answers.

The fixes in this run:

- **A response schema for each item.** Forced choice items require `choice` and restrict it to an `enum` of exactly the letters on offer, so a malformed or out of range letter is impossible rather than scored as wrong.
- **`{abstain:false, answer:null}` is now `schema_valid:false`**, not a confident wrong answer.
- **`confidence` is checked against its range**, and unknown fields are rejected, so `schema_valid` now means what it says.
- **Numbers written as words** (`"Four"`, `"eight"`) and trailing nouns (`"Sahara Desert"`) are scored correctly. The old bias worked in one direction only: it could only turn confident correct answers into confident errors, which is the worst direction for AUROC.
- **`total_ms` now includes time spent waiting in the queue**, as its comment always claimed.
- **Attempted now means not abstained**, so abstentions no longer leak into the AUROC.
- **Article stripping is greedy**, so equivalent phrasings converge.

## Two defective items found by the models

**All three** models missed two items, which far more often indicates a defective item than a shared blind spot. Both were removed:

- `fc-most-borders`: China and Russia **both** border 14 countries, so there is no single answer.
- `fc-nile-vs-amazon`: whether the Nile or the Amazon is longer is genuinely disputed.

Three other items that all models missed were **kept**, because they are the intended traps: `fc-human-bones-baby` (about 300, not 206), `fc-hundred-years-war` (116 years) and `fc-declaration-signed` (2 August, not 4 July).

**A confound introduced and then removed.** To remove bias from the canonical option order, each item's options were rotated by the item's index. That gave paraphrase twins a *different* option order from their originals, so screen #3 would have measured sensitivity to position while reporting sensitivity to paraphrase. Twins now inherit the rotation of their original. The problem was caught because twins changed *correctness*, not only confidence, in the first run.

## Next steps

1. **A bootstrap that accounts for clustering**, resampling items rather than records, so that permutations and twins stop inflating the precision. No confidence interval can be published until this is done.
2. **Larger unanswerable strata.** With 4 to 5 items each, the abstention metrics mean little.
3. **Coherent false premises absent from the corpus**, matched to the obvious ones on the type of falsehood and the form of the question. This is still the experiment that would make the coherence result mechanistic rather than suggestive.
4. **A second quantisation** of one model, which is the actual new contribution.

## Corrections

| Withdrawn or corrected statement | What is actually the case |
|---|---|
| The headline AUROC table (0.890 / 0.656 / 0.650) presented as *the* Type-2 AUROC | These values are **pooled**: 44 to 61% of the negative class consists of attempted unanswerable records, which is wrong by construction. `METRICS.md` calls this "screen 5 renamed". On answerable items only, the values are 0.849 / 0.643 / **0.566**, and with a cluster bootstrap two of the three models **fail** discrimination. |
| "25 to 41 errors per model … genuine" | The genuine errors on answerable items number 14 / 14 / 16 records on 10 to 11 items, and only 4 to 5 items are wrong under both orderings. |
| "All three models pass all five validity screens" | On the clean construct, only gemma4 passes, and its paraphrase screen sits at the threshold. |
| The coverage table, and "decline the bottom 80% … 94% against 69%" | This was an **artefact of file order**, caused by slicing inside groups of tied values. The 20% row ranged from 0.765 to 1.000 when the order was shuffled, and no threshold reaches 20% coverage. What can actually be achieved is confidence ≥ 1.0, giving about 66 to 75% coverage at about 0.91 to 0.93. |
| The "accidental stability check" (0.645 to 0.656 …) | No log from before the upgrade survives, because it was overwritten. gemma4 **failed** the paraphrase screen before the upgrade, and the phrase "barely moved" left that out. `backend_version` cannot attribute a record to a state of the driver and kernel. |
| granite `premise_challenged` on underspecified items: **40%** | The correct value is **20%**. The 40% came from the run before the upgrade; the table mixed two runs. |
| "trailing head-nouns (`Sahara Desert`) score correctly" | This is false. Only the fix for numbers written as words exists. |
| "Three other all-models-missed items were kept" | Only `fc-declaration-signed` was missed by all models. `fc-hundred-years-war` was 3/6, answered by **position A in every record**. |
| "granite states 0.991 while scoring 0.577" | These figures describe different populations. Compared like for like, the values are 0.961 against 0.577. |
| granite's obvious false premise value of **0.500** in the coherence table "separates cleanly" | Four attempted rows sit at about 1.0 and four abstained rows at 0.0. This is the abstention convention artefact that M1 §4 had already withdrawn. |
| `fc-fp-king-usa`, `fc-stale-tallest-building` | These items were defective. The true rejection of the premise was offered as an option, and the "stale" item had one defensible answer. Both were fixed or removed in 0.3.0. |
| The paraphrase screen values (0.079 / 0.003 / 0.000) | `_twin_pairs` mixed option orderings. The true values are 0.113 / 0.003 / 0.037. |
