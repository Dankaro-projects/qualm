# M1 validity screen, 2026-08-17

This report was corrected on 2026-08-17 after an adversarial review. Four independent reviewers challenged the first version, and the corrections that survived verification against the raw logs are listed in the "Corrections" section at the end. **The arithmetic in the first version was not wrong.** Every number it reported reproduces exactly from the logs. What was wrong was the interpretation of what the numbers measure, and the conclusions drawn from them.

**Run provenance.** 102 items × 3 models = 306 probes, with 0 harness failures. Items: `datasets/pilot.jsonl`. Prompt set `v1`, harness 0.1.0, all models **q4_K_M**, temperature 0, seed 0, `num_ctx` 8192, `think:false`, 4 concurrent requests.

To reproduce, **name the three files explicitly**. A `*-pilot.jsonl` glob also matches the quarantined stale log, and the analysis then crashes:

```bash
cd python && uv run python -m qualm_analysis.cli \
  ../results/raw/granite4-1-8b-pilot.jsonl \
  ../results/raw/qwen3-5-9b-pilot.jsonl \
  ../results/raw/gemma4-12b-pilot.jsonl
```

---

## 1. The headline metric cannot be estimated on this item set

This is the most important finding, and the first version of the report did not contain it.

**None of the three models makes a genuine knowledge error on any of the 46 answerable items.** Every answerable item scored as incorrect is either an abstention or a **scoring bug**:

| Model | scored wrong | actually |
|---|---|---|
| granite | `Four`, `eight` | **correct**: `parse_number` cannot read numbers written as words |
| granite | `Sahara Desert` | **correct**: `Exact` demands `Sahara` |
| gemma4 | `Naypyitaw` | **correct**: a spelling variant missing from the alias list |
| qwen | (one unnecessary abstention) | genuine, but not a knowledge error |

Type-2 AUROC asks whether confidence anticipates an error. With no errors, there is no negative class:

| granite4.1:8b | scoring as it stands | scoring corrected |
|---|---|---|
| **Pooled** (answerable plus attempted unanswerable) | 0.901 [0.785, 0.986] | **0.973** [0.940, 0.994] |
| **Answerable only** (the clean construct) | 0.601 [0.395, 0.956] | **UNDEFINED, 0 negatives** |

Both columns are correct, but they measure different quantities. The pooled figure, which the first version published as the headline, largely reflects the gap in confidence between *answering a real question* and *answering an unanswerable one*. That is **screen 5 (`unanswerable-separation`) under a different name**, not metacognition about correctness. On the answerable stratum, where the construct is clean, granite's lower bound is 0.395, so it **fails the discrimination screen's own criterion**. After the scoring is corrected, the value is undefined.

**Consequently, the pilot cannot measure Type-2 AUROC, and no change to the code can alter that.** The items are too easy. This is the reason for the change of format described in `docs/ROADMAP.md`.

## 2. What survives of the coherence result

The table gives the abstention rate on false premises that differ in how obviously false they are, with n, Wilson intervals and Fisher exact tests, as `docs/METRICS.md` requires:

| Model | obvious FP | coherent FP | Wilson 95% | Fisher p | dedup p |
|---|---|---|---|---|---|
| `granite4.1:8b` | 9/9 | **4/13 = 31%** | [13%, 58%] | **0.0017** | 0.0047 |
| `gemma4:12b` | 9/9 | 9/13 = 69% | [42%, 87%] | 0.115 | 0.242 |
| `qwen3.5:9b` | 9/9 | 13/13 = 100% | [77%, 100%] | 1.000 | 1.000 |

The *dedup* column excludes paraphrase twins, which are not independent observations.

**What survives:** `granite4.1:8b` abstains on every obvious false premise but on only 4 of 13 coherent ones (p = 0.0017), and the effect survives deduplication. That is a real effect in one model.

**What does not survive:**

- **The gemma4 row is withdrawn.** Its four answers that were not abstentions were `cfp-australia-capital-sydney` (`answer:null`), `cfp-pluto-ninth-planet-reinstated` (`answer:null`), `cfp-australia-capital-sydney-p` (`answer:null`) and `cfp-einstein-nobel-relativity`, which returned `"1921 (for the photoelectric effect)"`. Three produced no answer. The fourth is gemma4 **correctly rebutting the premise** and being scored `correct:false` for doing so. gemma4 accepted **zero** coherent false premises.
- **The qwen row carries no information**, because both arms sit at 100% against 100%, a ceiling.
- The pattern therefore holds in **1 of 3 models**, not 2 of 3.

### Coherence is probably not the mechanism

Of granite's 9 answers to coherent false premises, only **3 are confabulations** (`cfp-everest-tallest-base` returned 2715, `cfp-great-wall-visible` returned "approximately 10 kilometers", and `cfp-columbus-proved-round` returned "second voyage"). The others returned a **true related fact**: Einstein 1921 (correct), Edison 1879 (correct), Sydney 1908 and 1927 (both real dates), and goldfish "3 seconds" (the popular figure the item asked for).

The design cannot exclude three alternative explanations, and at least two of them are simpler than Kahneman's mechanism:

1. **The type of falsehood, not coherence.** All 9 obvious false premise items presuppose a **referent that does not exist** (Atlantis, unicorns, dragons, a square circle). All 13 coherent items presuppose a **false relation between real entities**. Detecting the first kind needs only knowledge of the entities, which is a much cheaper cue.
2. **Frequency in the training data.** All 13 coherent items are *famous misconceptions* (Einstein and relativity, Columbus and the round Earth, goldfish memory, the Great Wall seen from space). These falsehoods appear often in the training corpus. Kahneman's mechanism predicts credulity for *any* smoothly stated falsehood, including new ones; an account based on corpus frequency predicts it only for famous myths. This item set contains only famous myths, so it cannot tell the two apart.
3. **The form of the question.** 92% of coherent items ask for a number or a date, against 33% of obvious items. Restricted to numeric items, the Fisher test gives **p = 0.077** in this run.

**The claim "Kahneman's 2009 mechanism, measured" is withdrawn.** Coherence was never operationalised: no measure of coherence, rating or manipulation check was computed for any item. It is only a label in a JSONL file.

### The dependent variable changed after the fact

`docs/METRICS.md` registered the prediction in advance in terms of **confidence**: *"confidence should be high and flat, indistinguishable from genuinely answerable items."* On that measure the prediction **fails**. granite's confidence on coherent false premises is 0.578, against 0.999 on answerable items, so the two are separated rather than indistinguishable.

The result above uses the **abstention rate** instead. That change is *justified*, because the confidence field has two meanings (see §4). It is nevertheless a different test, and it was wrong to report it as confirming the prediction registered in advance.

## 3. Full behaviour table

Each cell shows the abstention rate and the mean stated confidence. **Read the confidence figures for the answerable rows only**; see §4.

| Category | n | `granite4.1:8b` | `qwen3.5:9b` | `gemma4:12b` |
|---|---|---|---|---|
| answerable | 46 | 0% / 1.00 | 2% / 0.97 | 0% / 1.00 |
| coherent-false-premise | 13 | 31% / 0.58 | 100% / 0.73 | 69% / 0.92 |
| false-premise | 9 | 100% / 0.00 | 100% / 0.76 | 100% / 1.00 |
| underspecified | 11 | 100% / 0.00 | 100% / 0.17 | 100% / 1.00 |
| unknown-answer | 11 | 100% / 0.00 | 100% / 0.86 | 100% / 1.00 |
| subjective | 6 | 83% / 0.15 | 83% / 0.88 | 100% / 1.00 |
| stale | 6 | 67% / 0.33 | 83% / 0.63 | 67% / 1.00 |

The quality of abstention, with Wilson intervals, is shown below. **No adjacent pair differs significantly** (qwen against gemma4, p = 0.271; gemma4 against granite, p = 0.197), so this table is not a ranking:

| Model | recall | Wilson 95% | precision | unnecessary abstention |
|---|---|---|---|---|
| `qwen3.5:9b` | 54/56 = 0.964 | [0.879, 0.990] | 0.982 | 0.022 |
| `gemma4:12b` | 50/56 = 0.893 | [0.785, 0.950] | 1.000 | 0.000 |
| `granite4.1:8b` | 44/56 = 0.786 | [0.662, 0.873] | 1.000 | 0.000 |

**These figures are not comparable to AbstentionBench.** Our unanswerable items are almost certainly easier, and `PROMPT_TEMPLATE` **explicitly permits abstention and lists the categories** ("unanswerable, underspecified, rests on a false premise, or has no single correct answer"), which the AbstentionBench prompt does not. This also puts the obvious false premise arm at a **ceiling of 100% with zero variance in all three models**, so the comparison in §2 sets a ceiling against a middle value. It is a different instrument, not evidence of better models. This caution applies to §2 as much as to this table.

## 4. Flaws in the instrument that this run exposed

**In prompt set v1, `confidence` has two meanings.** When the model answers, it is confidence in the answer; when the model abstains, it is confidence that declining was right. The models did not agree on a convention: granite writes **0.00** when abstaining, gemma4 writes **1.00**, and qwen uses **three different values** (0.0, 0.95 and 1.0). Mean confidence on abstained rows therefore cannot be compared across models.

**Three screen verdicts are artefacts of that field**, because screens 1 and 5 run on all rows, including abstentions:

- granite's **PASS** on `unanswerable-separation` (gap 0.813) comes largely from its 0.00 convention.
- gemma4's **FAIL** (gap 0.016) comes from its 1.00 convention. The statement that gemma4 is *"equally certain when declining and when answering"* is therefore **withdrawn**, because that number reflects the convention, not a finding. gemma4's AUROC on attempted items only, which is at chance, still supports the conclusion that it has no usable signal independently. The conclusion stands, but two of its three stated reasons do not.
- qwen's **FAIL** on `paraphrase-stability` (Δ0.390) breaks down into Δ0.010 on pairs it *answered* both times and Δ0.712 on pairs it *declined* both times. In other words, the meaningless field moved while the decision stayed the same. Meanwhile gemma4 PASSES at Δ0.100 while showing Δ0.143 on answered pairs. **On this screen, the order of PASS and FAIL is the reverse of the real behaviour.**

**A record of `{abstain:false, answer:null}` is counted as an answer.** Four such records exist, all in the coherent false premise stratum, which is the smallest stratum and the one the conclusions depend on most. These records should be marked `schema_valid:false` rather than scored as substantive answers.

**`schema_valid:true` does not mean the record passed schema validation.** It means only that three fields deserialised. A model that returned `confidence: 42` would be recorded as valid, and would push the variance screen to a PASS *for the wrong reason*.

## 5. Reproducibility

**The results are not reproducible at temperature 0 and seed 0.** Two granite runs exist, and they disagree on 5 to 7 items, **three of them inside the 13 item stratum behind the headline**. Abstention on coherent false premises is 23% in one run and 31% in the other; the drop is −77 or −69 points; the AUROC is 0.911 or 0.901. The likely cause is batch nondeterminism at `num_parallel:4`. **The first version reported no variance between runs and described the result as reproducible.** Repeated runs with the observed variance are now required before any number in this report may be quoted.

`results/raw/granite-pilot.jsonl` is a **stale log**. All 102 of its 102 records violate `run-record.schema.json`, because they have no `item` block, and they crash `cli.py`. The file carries the same `harness_version: 0.1.0` as the valid logs, because the version was not raised when the record shape changed. It has been moved to `results/raw/quarantine/` and is kept only as the evidence for the nondeterminism described above.

## 6. The screens, and why the overall verdict tells us nothing

| Screen | granite | qwen | gemma4 |
|---|---|---|---|
| variance | PASS | PASS (at the `min_distinct=4` boundary) | **FAIL** |
| discrimination | PASS (pooled) / **FAIL** (answerable only) | PASS | **FAIL** |
| paraphrase-stability | PASS | **FAIL**, but see §4 | PASS, but see §4 |
| order-stability | **not run** | **not run** | **not run** |
| unanswerable-separation | PASS, but see §4 | PASS | **FAIL**, but see §4 |

All three models read NOT USABLE, but `order-stability` **never ran**, because the pilot has no permutations of multiple choice options. The change of format also fixes this. For granite, screens 2 and 5 also measure **the same contrast**, so they do not corroborate each other independently. The overall verdict should not be quoted.

## 7. What this changes

1. **The pilot cannot measure Type-2 AUROC.** This is not a bug: the items are too easy, and there is no negative class. A change of format is required.
2. **meta-d′ and M-ratio are undefined for open questions** (arXiv 2603.25112, the paper that `docs/RESEARCH.md` cites for them). This is another reason the format must change.
3. **The coherence probe is worth keeping**, but it must be rebuilt: falsehoods absent from the training corpus, matched on the form of the question and the type of falsehood, and a `premise_challenged` field in the response so that a rebuttal is not scored as credulity.
4. **Rescore rather than rerun.** The raw answers are preserved, so the scoring bugs can be repaired without running the models again.
5. `gemma4:12b` is still the weakest of the three on this measure (its AUROC on attempted items only is at chance), and still the best at producing valid JSON. The two measures are unrelated.

## Corrections

The first version of this report overclaimed. It is not included in this repository. The following statements from it were withdrawn:

| Withdrawn statement | Reason |
|---|---|
| "The coherence hypothesis is confirmed" | It holds for **one** of three models. The gemma4 row was an artefact (see §2). |
| The `gemma4:12b` coherence row (−31 points) | Three of its four "answers" were `answer:null`, and the fourth was a **correct rebuttal** scored as credulity. Fisher **p = 0.115**. |
| "confidently inventing a year for Einstein's relativity Nobel" | granite answered **1921**, which is the correct year. |
| "a date on Shakespeare's birth certificate" | granite **abstained** on that item. |
| "same domain mix" | This was false. Mathematics was 2/9 against **0/13**, and history 1/9 against **5/13**. |
| "clean, novel, **reproducible** result" | A second granite run in this repository disagrees on 3 of the 13 headline items. |
| "M1 passes as a kill gate; the signal is real in 2 of 3 models" | Type-2 AUROC **cannot be estimated** on the clean construct. See §1. |
