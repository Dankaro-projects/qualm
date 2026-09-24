# Estimating the risk of acting on an answer

**Status: proposed evaluation, 24 September 2026.** Qualm currently measures signals and applies a rule-based gate. It does not produce a calibrated probability of error or an estimate of the consequences of acting. This plan tests whether those estimates would improve one real document-based workflow before changing the runtime.

The first candidate task is answering a factual question from a specified document collection, with evidence, for use in a professional report. The workflow owner, collection and exact downstream action still need to be selected. Existing SQuAD 2.0 results establish a baseline experiment, not acceptance for this workflow.

## 1. Define the event and the decision

The unit of evaluation is one question, the evidence available at the decision time, one proposed answer and one specified downstream action. An answer is **unfit for the action** if it is incorrect or lacks the support required by that action. A factually true answer can still be unfit when the workflow requires documentary evidence that is absent.

Keep the following quantities separate:

| Quantity | Definition | Intended use |
|---|---|---|
| Error probability, `p_error` | Estimated probability that the proposed answer is unfit, given the recorded signals and workflow | Identify answers that need checking |
| Consequence | Cost of acting on an unfit answer, defined by the workflow owner | Distinguish a correctable draft error from a consequential external action |
| Expected loss | Probability-weighted cost of each available action | Compare accepting, reviewing and declining |

Do not call disagreement, token probability, AUROC or stated confidence an error probability. Calibration must establish that relationship on the target task. A probability is an estimate for comparable cases, not proof about an individual answer.

For an initial binary-cost model, where an acceptable answer has zero additional loss:

```
loss(accept)  = p_error * cost_of_wrong_action
loss(review) = review_cost + review_delay_cost + expected_loss_after_review
loss(decline)= expected_cost_of_not_acting
```

All costs must use comparable units and a declared time horizon. Where error consequences vary, estimate loss over the relevant outcomes rather than multiplying by one average cost without justification. Review can still produce an error or a decline; measure those outcomes. Declining can avoid harm while losing a useful opportunity. If costs cannot yet be estimated, report a sensitivity analysis over explicitly hypothetical scenarios instead of claiming a business benefit.

The owner defines the maximum acceptable error rate, minimum useful acceptance coverage and review capacity before the final evaluation. A low expected cost does not override a required approval or an unacceptable consequence. Missing evidence or an unsupported model/workflow configuration must not become a low-risk default.

## 2. Establish evidence and labels

Use the selected workflow's real case distribution for the main evaluation. Keep a separate challenge set covering absent answers, false premises, conflicting or stale sources, incorrect entity/date relationships, and answers that are consistently wrong across samples. Do not pool deliberately oversampled failures into a headline operational error rate without justified weighting.

For each case, preserve the question, proposed action, document identifiers and versions, evidence supplied to the model, proposed answer, runtime signals and the eventual disposition. If retrieval is involved, distinguish an answer absent from the collection from one present in the collection but missing from the retrieved passages. Neither condition is established by retrieval scores alone.

Reference labels must record answerability, factual correctness, whether the cited evidence supports the requested relationship, and fitness for the specified action. Use defined categories for failure reasons, with short evidence notes where needed. An answer span appearing in a document is not enough: a construction date does not establish an acquisition date. Human reference labelling is allowed; the earlier machine-only benchmark constraint does not establish ground truth for professional actions.

Reviewers label cases without seeing the proposed risk score. Independently adjudicate disagreements and consequential cases, record unresolved cases separately, and disclose their number and exclusion rule. Judge-model output may be an evaluated signal, but cannot be the sole reference label for its own evaluation.

Partition by document family or underlying case before fitting or choosing thresholds. Near-duplicates, related questions and repeated samples stay in the same partition. Use development data for feature and estimator choices, calibration data for probability mapping and policy thresholds, and an untouched test set for the final comparison. A later time period provides a further check for drift. Repeated generations do not count as new independent cases.

Choose sample size from the required uncertainty at the intended operating point. A few hundred total questions may produce too few accepted answers or errors to validate a low-risk claim. For illustration, even zero errors among 100 independent accepted cases gives a one-sided 95% binomial upper error bound of about 3%; it does not establish a 1% limit. Correlated cases require a method that accounts for their grouping, not that binomial calculation.

## 3. Compare observable signals before fitting a risk estimate

Start with signals Qualm already records: answer agreement, verbatim passage occurrence, malformed responses, abstention and stated confidence. The existing `grounded` field means text occurrence only. Preserve its meaning; a new evidence-support check must be separately named and versioned.

Measure evidence support and answerability checks as additions. If exact normalised string matching treats equivalent answers as disagreement, compare a semantic-agreement method separately and measure its own mistakes and inference cost. Research on [semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) motivates this comparison; it does not establish that the method will work on this collection.

Compare these policies on the same cases and candidate-answer procedure:

1. The current workflow and an ungated model-answer baseline, retaining explicit declines and transport/format failures.
2. The existing Qualm gate with its complete configuration recorded, plus a passage-occurrence-only policy.
3. Agreement plus evidence-support and answerability checks, with ablations showing each check's contribution.
4. The simplest calibrated estimator that adds value over those rules.
5. An independent reviewer or second model only if its incremental benefit justifies its cost.

The current configured gate includes a stated-confidence threshold as well as unanimity and optional passage checking. It is not identical to the report's grounding-and-unanimity ablation. A second-model comparison must include that model answering alone, so gains from using a stronger or more cautious model are not mistaken for gains from the gate.

Use only information available before the action as estimator inputs. Never include the reference answer, adjudicated label, later outcome or post-decision review. Start with empirical error rates for adequately supported signal groups or a small regularised model selected on development data. Do not build a general scoring framework or force a scalar score if a simple rule performs as well. Sparse or out-of-scope cases remain unvalidated, not zero risk.

## 4. Evaluate probability quality and decision value separately

Freeze the candidate procedure, estimator, thresholds and cost assumptions before opening the test set. Keep a manifest of model digest, quantisation, prompts, sampling settings, retrieval/document versions, code revision, split identifiers and estimator version. Existing logs remain readable; any later format change needs its own version and validation.

| Measure | Required reporting |
|---|---|
| Calibration | Reliability table/plot, Brier score, bin counts and uncertainty for `p_error`; compare with a constant error-rate estimate fitted without test labels |
| Discrimination | AUROC for predicting unfit answers, with explicit label direction; keep answerable-only Type-2 AUROC separate |
| Selective risk | Wrong accepted answers / all accepted answers, at each frozen operating threshold; zero accepted answers means undefined risk |
| Coverage | Accepted / all eligible cases, alongside review, decline and failure counts |
| Operational cost | Inference calls, latency, reviewer time, review capacity and residual errors after review |
| Decision value | Paired difference in total observed loss versus the current workflow and simplest useful gate, with cost sensitivity and uncertainty |

Report all denominators, independent case/document groups and intervals. Bootstrap paired differences at the independent group level, keeping related items together. Report results by failure category and consequence where sample sizes support them. Unknown outcomes remain visible; they are not successes. No pooled accuracy figure should conceal abstentions or a large review burden.

Advance only if the frozen policy meets the owner's error, coverage and review-capacity criteria on held-out data and improves expected loss over the relevant baseline with adequate evidence. If uncertainty is too wide, the result is inconclusive. If a simple rule performs equally well, keep the rule. If no policy is useful, publish the negative result.

## 5. Run in shadow mode before allowing automated action

After the offline criteria pass, evaluate on fresh workflow cases while the existing decision process remains authoritative. Record what Qualm would have accepted, reviewed or declined and compare with adjudicated outcomes. This observes the real system boundary without allowing an unvalidated score to act on it.

Measure actual review time and outcomes rather than assuming a perfect reviewer. Sample accepted cases for audit, because observing only reviewed cases hides errors that the gate lets through. Record sampling probabilities and account for them when estimating the overall error rate. Freeze the policy during each evaluation window.

A change in model, prompt, quantisation, retrieval, source mix or action can invalidate calibration. Such a change requires a new evaluation or an explicit check that the existing evidence still applies. Deployment is a later decision based on the shadow report; this plan does not authorise automated publishing or other consequential actions.

## 6. Keep reward-driven overconfidence as a separate question

[Reward calibration research](https://arxiv.org/abs/2410.09724) reports that reward models can favour expressed confidence, and [research on evaluation incentives](https://www.nature.com/articles/s41586-026-10549-w) explains how accuracy scoring can reward guessing. These findings motivate scrutiny of confidence; they do not identify the cause of a particular Qualm error.

This evaluation tests whether observations predict action-relevant errors. It does not measure a model's motives or establish an internal experience of doubt. A causal claim about reward training would require a separate controlled comparison of training conditions or checkpoints. Rewarding the appearance of hesitation is not an acceptance criterion.

## Delivery order

The [roadmap](ROADMAP.md#action-risk-evaluation-planned) tracks four dependent steps: select the workflow and acceptance criteria; build labelled cases and signal baselines; calibrate and evaluate on held-out cases; then run a shadow trial and make a deployment decision. Each step must produce its stated evidence before the next begins. No risk estimator, new runtime field or production integration is implemented by this documentation change.
