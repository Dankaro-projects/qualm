# Research foundation

Every claim in this document was checked against its source on **2026-08-17**. Where the literature disagrees, the disagreement is stated rather than resolved. Where a paper's finding changes a design decision in this project, the decision is named.

The status of each source is marked as follows. **Verified** means that the abstract or the paper was read directly. **Contested** means that credible work argues the opposite.

---

## 1. Hesitation is the gap, and it remains unsolved

**AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions**
[arXiv 2506.09038](https://arxiv.org/abs/2506.09038) · Meta / FAIR · NeurIPS 2025
· [code](https://github.com/facebookresearch/AbstentionBench) · *verified*

- The benchmark covers 20 datasets and **35k+ unanswerable questions**. The categories are unknown answers, underspecification, false premises, subjective interpretation and stale facts.
- Across **20 frontier LLMs, abstention remains unsolved, and scaling helps very little.**
- **Fine-tuning for reasoning degrades abstention by 24% on average.** This includes mathematics and science, the domains for which reasoning models are explicitly trained.
- The mechanism is that models **invent the missing context** and then answer definitively.
- **Longer reasoning chains improve accuracy, but abstention does not improve.** More precisely, abstention worsens on UMWP and stays flat on GSM8k-Abstain. That is a result on 1 of 2 datasets, so describing it as "a genuine tradeoff" slightly overstates it. The direction is consistent, but it should not be cited as universal.
- A good system prompt helps in practice, but it does not fix the underlying inability to reason about uncertainty.

> **Design consequence.** More reasoning is not the lever. Any plan to fine-tune on reasoning traces in order to obtain better judgment targets the one intervention that has been measured to make the target behaviour worse. Our item sets should adopt the AbstentionBench *taxonomy* of unanswerability, because its four categories are the right axes along which to stratify.

---

## 2. Quantisation breaks the obvious metric

This is the finding on which the project is built.

**Quantisation Reshapes the Metacognitive Geometry of Language Models**
[arXiv 2604.08976](https://arxiv.org/abs/2604.08976) · *verified*

- The paper compares Llama-3-8B-Instruct at **Q5_K_M and at f16**.
- **Domain level M-ratio profiles show *zero* correlation between the two formats.** Calibration is reshuffled by domain: some domains improve and others degrade.
- **Type-2 AUROC profiles are perfectly stable across the formats.** The underlying discrimination survives; what changes is the *normalisation* of confidence.
- A training intervention that amplified confidence **failed** to improve metacognition.

> **A weakness in the evidence.** The values ρ = 0.00 and ρ = 1.00 are Spearman correlations over **n = 4 knowledge domains**. With n = 4 there are only 24 possible orderings, so neither value is statistically significant, and both are unremarkable outcomes at that sample size. The figures are quoted accurately here, but this project relies heavily on the citation, and n = 4 cannot carry that weight. **The choice of AUROC over M-ratio is therefore well motivated but not established.** Replicating the effect at q4 is itself part of what M3 would contribute.
>
> **Design consequence.** This is the central one. We run **q4_K_M**, which is more aggressive than the quantisation tested in the paper. Therefore:
> 1. **AUROC is the headline metric.** M-ratio is reported, but never compared across quantisations.
> 2. **Quantisation is a primary experimental variable**, recorded on every row.
> 3. This is also the *contribution*: if frontier profiles do not transfer to quantised weights, the local stack needs its own measurement.

---

## 3. Measuring it properly

**Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory** [arXiv 2603.25112](https://arxiv.org/abs/2603.25112) · *verified*

- The study runs **224,000 factual QA trials on 4 models**.
- Metacognitive efficiency is **dissociated from accuracy**: it varies by a factor of about 1.98 and is "not predicted by accuracy".
- The paper does **not** support meta-d′ and M-ratio for open-ended questions. In its own words, the *"meta-d' efficiency ratio **is not well defined for open-ended QA**, which lacks a two-alternative Type-1 decision"*. It therefore uses a model-free measure, normalised metacognitive information (meta-I₂ᵣ), instead.

> **Design consequence.** This is a constraint on the design, not a footnote. `datasets/pilot.jsonl` contains open-ended factual questions with no two-alternative Type-1 decision, which is exactly the case for which the paper says meta-d′ is undefined. The estimation plan in [METRICS.md](METRICS.md) therefore cannot be carried out on that item format. It is one of two reasons why the benchmark moved to forced-choice items. The other reason is that the open-ended pilot produced **zero** errors on answerable items, which left Type-2 AUROC without a negative class as well.
>
> The M-ratio evidence on which the project actually relies comes from §2 above, by the same author, in a setup close to a two-alternative decision.

The definitions we use are listed below. [METRICS.md](METRICS.md) gives the operational versions.

| Quantity | Meaning |
|---|---|
| d′ | Type-1 sensitivity: how much the model knows |
| meta-d′ | The d′ that an ideal observer would need in order to produce the observed relationship between confidence and correctness |
| M-ratio | meta-d′/d′: efficiency, normalised for task difficulty. 1 is optimal; <1 indicates under-monitoring |
| Type-2 AUROC | How well confidence alone separates correct from incorrect answers, independent of any threshold |

**Screening before interpretation.**
[arXiv 2604.17714](https://arxiv.org/pdf/2604.17714) and the companion papers
(2604.17716, 2604.17707) · *verified (abstracts)*

- These papers propose a portable validity protocol for confidence signals derived from benchmarks.
- They argue that confidence profiles must be **screened for validity before they are interpreted**. Otherwise an analysis produces confident conclusions about a signal that was never present.

> **Design consequence.** The validity screen is **M1**, and it runs before any leaderboard.
>
> These three papers do not discuss whether M-ratio is sensitive to format, and they do not recommend metrics that are stable across formats. That recommendation, naming both AUROC₂ and the NLP gap, comes from **§2 (arXiv 2604.08976)**.

---

## 4. The signal exists but goes unused

This is the result on which the harness is based.

**LLMs Know When They Know, but Do Not Act on It: A Metacognitive Harness for Test-time Scaling** [arXiv 2605.14186](https://arxiv.org/abs/2605.14186) · *verified*

- The work is grounded in the metamemory theory of **Nelson and Narens**.
- It elicits two signals: **FOK** (feeling of knowing), which is prospective and given *before* solving, and **JOL** (judgment of learning), which is retrospective and given *after* solving.
- Both signals correlate meaningfully with actual correctness, **but models do not allocate more reasoning effort to items on which they have low confidence.** The signal is produced and then discarded.
- The harness turns the two signals into **three control actions: trust the answer, retry with feedback, or aggregate several attempts.**
- **Pooled accuracy rises from 48.3 to 56.9 on Claude Sonnet-4.6, with no parameter updates** and no fine-tuning for any benchmark. The evaluation uses HLE-Verified, LiveCodeBench v6 and R-Bench-V.

> **Design consequence.** This is close to the architecture of `qualm-harness`. The central point is that **the gain comes from reading the signal externally, not from training the model to state it better.** The project can therefore deliver value as a harness, without any fine-tuning.

---

## 5. Why our choice of models works against us

**Mind the Confidence Gap: Overconfidence, Calibration, and Distractor Effects in LLMs** [arXiv 2502.11028](https://arxiv.org/abs/2502.11028) · *verified*

This paper studies **prompting with added distractors**. It covers 9 models and 3 factual QA datasets, scored by ECE and accuracy, and it does not stratify results by the difficulty of individual items.

In its own words: *"large RLHF-tuned models display inherent calibration strengths but can paradoxically suffer **increased miscalibration on easier queries**, whereas smaller models benefit disproportionately from distractor prompts but remain significantly miscalibrated."*

- Large models become *worse* on easy queries. The paper attributes this to confidence inflated by the multiple-choice context, **not** to a pattern that varies with difficulty.
- Smaller models remain miscalibrated, but they **do** respond to the intervention, which is itself a form of variation.

**Xiong et al. (2024)**, on eliciting confidence, find that verbalised confidence is consistently overconfident and clusters in the **80 to 100%** band.

> **Design consequence.** This rests on Xiong et al. A **larger control model** is needed, for a modest reason: without one, we cannot tell whether small models are poor at this task or whether the elicitation method is poor. That reason stands on its own. The paper 2502.11028 should not be cited for an interaction between model size and item difficulty, because it does not report one.
>
> Raw verbalised confidence may need rescaling, but the **failed** amplification result in §2 suggests that this will not be easy.

---

## 6. Intuition is the component that is not missing

**Kahneman & Klein (2009), "Conditions for Intuitive Expertise: A Failure to Disagree"** · *American Psychologist* 64(6), pages 515 to 526 · *verified*

This paper is an adversarial collaboration between researchers in heuristics and biases and researchers in naturalistic decision making. It describes skilled intuition as **recognition**, following Simon, and states that it requires **two** conditions:

1. an environment that is **regular enough** to contain valid cues that can be learned;
2. **prolonged practice with rapid feedback of high quality**, which the authors describe as "tens of thousands of hours".

Where either condition is missing, experience produces **confident judgment without accurate judgment**, and subjective confidence does not reliably indicate accuracy. Kahneman's mechanism is that confidence follows the **internal coherence of the evidence** rather than its quality, so evidence that is thin but consistent produces the most overconfident judgments.

> **Design consequence.** Two points follow, and they reverse the usual framing.
> 1. **A transformer consists entirely of intuition**: it completes patterns over a corpus, which is recognition *without* the validity check. The task is not to build intuition but to constrain it.
> 2. **Confidence driven by coherence is an accurate description of next token prediction over a retrieved context.** Kahneman described this failure mode in 2009. It gives a hypothesis that the benchmark can *test*: if confidence tracks coherence rather than correctness, confidence should be **high and flat** on premises that are internally consistent but false. The false premise category of AbstentionBench is the natural probe. **This was the most interesting original experiment available to the project**; §8 records how prior work has since narrowed it.

---

## 7. Whether any of this is real remains contested

*Contested. Neither side should be presented as settled.*

- **In favour:** frontier models show at least rudimentary metacognition, in that they detect an internal confidence signal and act on it. Their sensitivity is weak to moderate but real, and it varies by model family at equal task performance.
- **Against:** other work argues that LLMs lack metacognition entirely. On this view they have no internal monitors for conflict or for the likelihood of error, and "I'm not sure" is a linguistic construction rather than a reading of an internal state. Scholten et al. describe **"metacognitive myopia"**: the absence of monitoring and control processes, which produces systematic bias.
- **Ackerman (2025)**, using the Delegate Game and Second Chance Game paradigms, finds metacognition that is **limited and dependent on context**.

The following surveys and test batteries are useful for their methods:
- [Metacognition in LLMs: Foundations, Progress, and Opportunities](https://arxiv.org/abs/2607.11881) (Yale NLP) · [repo](https://github.com/yale-nlp/LLM-Metacognition). This is the current overview of the field.
- [The Metacognitive Monitoring Battery](https://arxiv.org/pdf/2604.15702). A benchmark across domains whose task structure is worth adopting.
- [Domain-level metacognitive monitoring: a 33-model atlas](https://arxiv.org/pdf/2605.06673). The frontier baseline against which our local results are compared.

> **Design consequence.** We do **not** take a side. We build the validity screen so that the data can decide. If the screen shows that the signal is manufactured, that is a publishable null result, and the harness will be built on a different signal.

---

## 8. Prior work that limits the claim of novelty

These papers were found during a review of the project on 2026-08-17. Each was fetched from its source during the review and fetched again by hand before being recorded here.

**Verbal Confidence Saturation in 3-9B Open-Weight Instruction-Tuned LLMs: A Pre-Registered Psychometric Validity Screen**, Cacioli, [arXiv 2604.22215](https://arxiv.org/abs/2604.22215) · *verified*

The study tests seven instruction-tuned open-weight models of 3 to 9B parameters, from four families, at **Q5_K_M GGUF on consumer hardware**. It uses 524 TriviaQA items, numeric (0 to 100) and categorical verbalised confidence, greedy decoding and 8,384 trials, and it reports AUROC₂ and a validity classification for each model. **All seven models failed the validity screen on numeric confidence**, with a mean ceiling rate of 91.7%. Categorical elicitation disrupted task performance in six of the seven models. Token log probability predicted verbalised confidence with a cross-validated R² < 0.01. **The study contains no unanswerable items.**

> **Consequence.** This paper covers most of what this project originally set out to claim. The remaining gap is narrower: models from the 2026 generation, q4 rather than q5, Ollama as the runtime, and unanswerability strata with a field for rejecting a false premise. Our finding that confidence collapses onto 3 to 8 distinct values, and that two of three models perform at chance, **replicates** this paper's result on the next generation of models. That is worth stating, but it is not novel.

**Two Axes of LLM Abstention: Answer Correctness and Question Answerability**, Wagner, [arXiv 2607.08456](https://arxiv.org/abs/2607.08456) · *verified*

The study tests five instruction-tuned models of 2 to 14B parameters, from three families, on natural false premise questions (CREPE). In its own words: *"Ordinary answer-confidence tracks whether an answer is right but is nearly blind to whether the question is answerable; a linear probe on hidden states does the reverse."* Answer confidence, P(IK), P(True) and asking the model directly whether the premise is false all remain near chance on false premise items. The paper proposes a policy that certifies both axes.

> **Two consequences.** First, the pattern we observed for granite, where confidence could not be distinguished between answerable items and coherent false premise items, is this result, already generalised across five models. The project therefore no longer claims a coherence mechanism. Second, the paper explains *why* it is wrong to pool attempted unanswerable items into the negative class for AUROC: doing so mixes two axes that the signal separates in very different ways.

**Distilling Self-Consistency into Verbal Confidence**, Cacioli, [arXiv 2604.24070](https://arxiv.org/abs/2604.24070) · *verified*

On Gemma-3-4B-it, verbal confidence reaches an AUROC₂ of **0.554** at baseline, while self-consistency over ten samples reaches **0.999**. Verbalised confidence from a single pass is the weakest signal available on a small local model.

> **Consequence.** The field `signals.agreement`, which records self-consistency across permutations of the answer options, should be filled in and reported beside the verbalised AUROC. It is expected to perform much better, it only requires more permutations, and it is the number a gate should use.

The frontier 33-model atlas ([arXiv 2605.06673](https://arxiv.org/abs/2605.06673)) uses 1,500 MMLU items and still reports a median confidence interval width of about 0.2 for each domain. **A set of 58 authored items cannot support any statement at the level of a domain**, and it cannot be compared with the atlas at all unless MMLU items are adopted.

Other prior work narrows the claim in the same direction. arXiv 2604.08976 (§2) measured M-ratio, meta-d′ and Type-2 AUROC on a GGUF quantisation. arXiv 2607.10855 measured calibration and uncertainty at 2, 3, 4 and 8 bits across six quantisation methods, with reliability peaking at 4 bits. arXiv 2505.23854 covered 80 models, including quantised variants, with selective classification on MMLU-Pro. arXiv 2604.24070 reported verbalised confidence AUROC₂ on a small local model.

---

## Summary of implications

1. **Build for hesitation.** Reasoning is not the gap, and intuition is already oversupplied.
2. **Score with AUROC**, on answerable items only, with cluster confidence intervals and two runs. M-ratio is dropped (§3 and METRICS.md).
3. **Run the validity screen first.** Nothing can be interpreted until it passes, and on the clean construct only one of the three models passes.
4. **The harness needs no fine-tuning** to be worth building (§4).
5. **Do not add reasoning** to improve abstention (§1). It has been measured to make abstention worse.
6. **The large control model has been replaced** by gemma4 at q8_0, which gives a pair within one model. A cloud model is outside the claim (§5, ROADMAP M2).
7. **The coherence experiment is no longer a claim about mechanism.** Its categories remain as strata (§8).
8. **Measure self-consistency beside verbalised confidence** (§8). It is the cheapest result that can be cited, and it is the number a gate needs.
9. **Present the work as a technical report, not a paper** (§8). The gap in the leaderboard is narrower than two earlier versions of these notes stated.

---

## Corrections

The following corrections were made on 2026-08-17.

- §3 previously cited arXiv 2603.25112 as the authority for meta-d′ and M-ratio. The paper states the opposite for open-ended questions.
- §3 previously attributed to arXiv 2604.17714 and its companions a warning about the format sensitivity of M-ratio. That recommendation belongs to arXiv 2604.08976.
- §5 previously stated that arXiv 2502.11028 found larger models underconfident on easy items and overconfident on hard ones, and smaller models overconfident at every level of difficulty. Neither statement is in the paper, and the first is close to the reverse of its finding.
- Earlier notes claimed that nobody had measured the quantised local stack. The prior work in §8 shows that claim was false.
