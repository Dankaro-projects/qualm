# Metrics

Read §2 and §3 of [RESEARCH.md](RESEARCH.md) first. The choices in this document follow from evidence rather than preference, and the reasoning is recorded so that nobody later simplifies them away by mistake.

## A precondition that comes before everything else

Type-2 AUROC requires a **negative class that is not empty**, that is, items the model answers *wrongly*. On the M1 pilot, all three models made **zero** genuine errors across 46 answerable items, so AUROC was **undefined**, not merely weak. A pooled figure over answerable items *and* attempted unanswerable items can still be computed, and it looks excellent (0.973). However, it measures the gap between answering a real question and answering an unanswerable one, which is screen 5 under another name. This was learned on 2026-08-17.

**Always report the number of errors next to an AUROC.** A model that answers every item in a set correctly cannot show whether its confidence anticipates failure.

**Report the figure over answerable items only as the AUROC, and label the pooled figure beside it.** The command line tool reports both. The first report on forced-choice items quoted only the pooled figure, although 44 to 61% of its negative class consisted of attempted unanswerable records, and it had to be corrected. In addition, report results **from two runs**, because gemma4 and granite change 6 to 10 of 114 records between consecutive runs at temperature 0. Report **cluster confidence intervals** as well, because the permutation records and paraphrased twins of one item form one observation, not k observations.

## The headline metric: Type-2 AUROC

**Definition.** Given a set of attempts, each with a confidence value and a label for correctness, Type-2 AUROC is the probability that a randomly chosen *correct* attempt received higher confidence than a randomly chosen *incorrect* one.

- `0.5` means that confidence carries no information about correctness.
- `1.0` means that confidence separates right from wrong answers perfectly.
- `<0.5` means that confidence is **inversely related** to correctness. That is worse than no signal, and it is a real outcome that should be reported prominently.

**Why it is the headline metric.** It does not depend on any threshold. It does not matter whether a model says 0.9 when it means 0.6; only the *ordering* matters. This property is what allows it to survive quantisation: Type-2 AUROC profiles are **perfectly stable** between f16 and Q5_K_M, while M-ratio profiles show **zero** correlation across the same comparison.

We run q4_K_M. **A metric that changes when the model is quantised again measures the quantisation, not the model.**

## The secondary metric: M-ratio, always with a warning

**M-ratio = meta-d′ / d′.** It measures metacognitive *efficiency*, normalised for how much the model actually knows. `1.0` is optimal, and `<1.0` indicates under-monitoring.

We report it because it is the common measure in the field and reviewers expect it. Three rules apply without exception.

1. **Never compare an M-ratio across quantisation levels.** This applies between q4 and q8, and against a published f16 figure. Such a comparison is not valid.
2. **Always print the quantisation tag next to the value.** It must sit next to the value, not in a caption elsewhere, so that a screenshot cannot lose it.
3. **Reproduce any claim about M-ratio across models in AUROC.** If the two disagree, AUROC takes precedence, and the disagreement itself is reported.

**Estimation.** Use an established implementation, either the hierarchical Bayesian approach in the style of HMeta-d or the standard maximum likelihood fit of meta-d′. Do not write your own. Report credible intervals, because point estimates of meta-d′ on a few hundred trials are noisy enough to mislead on their own.

> **Removed from the roadmap on 2026-08-17.** No decision depends on it, it needs hundreds of trials per cell that this project will not have, and its instability across quantisations is already published. This section remains as the record of the reasoning, in case a reviewer asks for it.

> **A precondition added on 2026-08-17: meta-d′ requires a two-alternative Type-1 decision, and open-ended questions do not provide one.** arXiv 2603.25112, the paper that RESEARCH.md cites for this method, states that meta-d′ "is not well defined for open-ended QA" and uses a model-free measure (meta-I₂ᵣ) instead.
>
> `datasets/pilot.jsonl` is open-ended. **None of this section can therefore be computed on it**, and that is one of the two reasons why the benchmark moved to forced-choice items. The other reason is that the open-ended pilot produced zero errors on answerable items, so Type-2 AUROC had no negative class either.
>
> On forced-choice items, both measures are well defined. If an open-ended set is ever scored again, use meta-I₂ᵣ or AUROC₂, not M-ratio.

## Calibration: ECE and Brier score, reported but not headlined

ECE and the Brier score answer a *different* question: whether a stated 0.9 corresponds to 90% accuracy. They are exactly what quantisation disturbs. They are useful for the harness, because a gate needs a threshold in real units, but **not** for claims across models or across quantisations.

**Reliability diagrams are the most transparent presentation.** A single ECE figure hides whether a model is uniformly overconfident or badly wrong in one bin.

## Abstention metrics

These metrics are distinct from calibration, and they address the question that matters for a product. They follow the framing of AbstentionBench.

| Metric | Question |
|---|---|
| Abstention recall | Of the items that *should* be declined, how many were declined? |
| Abstention precision | Of the items declined, how many *should* have been declined? |
| Over-abstention rate | How many answerable items were wrongly declined? This failure makes a product useless |
| Coverage and risk curve | What is the accuracy on retained items as a function of how many are declined? |

**For the harness, lead with the operating points.** These are the coverage and the accuracy on retained items **at each confidence threshold that actually occurs** (`operating_points`). Do not use a coverage curve by percentile. These models produce only 3 to 8 distinct confidence values, so a point such as "20% coverage" falls inside a group of tied values, and the accuracy on retained items then depends on the order of the file. The first M3 report published such a row: it ranged from 0.765 to 1.000 when the records were shuffled, and no threshold could reach 20% coverage. A gate is a threshold, so report thresholds.

## The validity screen runs before anything above is interpreted

As §3 of [RESEARCH.md](RESEARCH.md) explains, a confidence profile cannot be interpreted until the signal has been shown to be a real signal. This is **M1** in the roadmap, and it comes before any leaderboard.

Each of the following minimum conditions must be met.

1. **Variance.** Confidence must vary at all. In our seed data, 36/36 samples fell in `[0.73, 0.98]`. A field that is almost constant cannot discriminate anything, yet it still produces a plausible AUROC near 0.5 that people read too much into.
2. **Discrimination above chance.** Type-2 AUROC must be significantly > 0.5, as shown by a bootstrap confidence interval, not by inspecting a point estimate.
3. **Stability under paraphrase.** The same item, reworded, should receive the same confidence. Confidence that changes with the wording is measuring the prompt.
4. **Stability under reordering.** For multiple-choice items, shuffle the options. Sensitivity to position is a known confound.
5. **Different behaviour on unanswerable items.** If confidence is the same on answerable and unanswerable items, there is no hesitation to measure.

**A model that fails the screen is reported as failing it.** That is a result, and possibly the most important one, because one line of the literature predicts exactly this (§7 of [RESEARCH.md](RESEARCH.md)).

## The coherence hypothesis

This was intended as the project's original experiment. It follows from Kahneman's mechanism (§6 of [RESEARCH.md](RESEARCH.md)): confidence is driven by the **internal coherence** of the evidence rather than by its quality.

**Prediction.** On false premise items that are internally consistent, confidence should be **high and flat**, and it should not differ from confidence on genuinely answerable items, because the model's evidence is coherent but wrong.

**Test.** Divide the item set into three strata: (a) answerable, (b) unanswerable in an obvious way, and (c) based on a false premise that is coherent. If confidence separates (a) from (b) but *not* from (c), that is direct evidence that the signal tracks coherence rather than correctness. It would also explain the failure mode in terms that a strategy audience already understands.

The test is inexpensive to run once the item sets exist.

> **No longer a claim about mechanism, as of 2026-08-17.** Our own evidence came from 1 of 3 models and 5 items. The granite contrast of 0.500 on obvious false premises is an artefact of the convention for scoring abstentions, which was withdrawn in §4 of the M1 report: four rows sit at about 1.0 and four at 0.0. The type of falsity, the frequency in the corpus and the form of the question were never controlled. arXiv 2607.08456 has since shown, across five models of 2 to 14B parameters, that answer confidence is almost blind to answerability on natural false premise data. The `false-premise` and `coherent-false-premise` strata remain in the leaderboard as strata, and this section remains as the record of the idea.

## Reporting rules

- **Give bootstrap confidence intervals for every headline figure.** Point estimates on a few hundred items invite over-interpretation.
- **Give n on every row.** Always record both n and the number of clusters (items), which is what `Estimate` prints.
- **Mean confidence over abstained rows reflects a convention, not a signal.** granite writes 0.0 on every abstention, while gemma4 varies. Any contrast that includes abstained rows must say so; the decline rate is the quantity that can be compared.
- **Publish null results.** A model with an AUROC of 0.5 belongs in the table. Selective reporting is the characteristic failure of this field.
- **Place the quantisation tag next to every calibration figure**, as the rules above require.
