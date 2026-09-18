# qualm

A qualm is a feeling of doubt, especially one that a person fails to act on.

Language models produce a usable signal about whether they know an answer, and then ignore it. `qualm` measures that signal on quantised models that people run on their own hardware, and provides a harness that acts on it.

The project has two parts that share one codebase:

- **A benchmark** that measures Type 2 AUROC, the ability of a model's confidence to separate its right answers from its wrong ones, for q4_K_M models served by Ollama.
- **A harness** that uses the signal at run time to accept an answer, send it for review or decline it.

## What this project adds

The metacognition of frontier models is well studied: a 33 model atlas and a monitoring battery across domains already exist. Quantised models have been studied too. arXiv 2604.08976 measured Q5_K_M, and arXiv 2607.10855 measured calibration at 2, 3, 4 and 8 bits. arXiv 2604.22215 ran seven open weight models of 3 to 9 billion parameters at Q5_K_M GGUF on a 16 GB consumer GPU, with verbalised confidence, a validity screen and AUROC₂; all seven failed the screen, and the study had no unanswerable items. arXiv 2607.08456 covers false premises and answerability across five models of 2 to 14 billion parameters.

The remaining gap is narrower: models of the 2026 generation, q4 rather than q5 quantisation, Ollama as the serving layer, and forced choice questions stratified by the reason a question cannot be answered, with a field for rejecting a false premise. That gap supports a technical report rather than a research paper.

The motivation is that quantisation changes calibration profiles at the level of domains while discrimination stays stable, so profiles measured on frontier models may not transfer to a 16 GB consumer GPU. The evidence for that effect is limited: it rests on a correlation across 4 domains (see `docs/RESEARCH.md`, section 2).

On the harness side there is a strong published result. Eliciting a confidence signal before and after solving, and using it to control inference, raised pooled accuracy from 48.3 to 56.9 without any change to the model's parameters. The signal exists and is not being used.

## The thesis

People name four qualities when they say a model is not human enough: judgment, reasoning, hesitation and intuition. Of these, only hesitation is both missing and measurable. Reasoning is largely solved. Models have too much intuition rather than too little, in the sense of pattern completion without the conditions that Kahneman and Klein describe for valid intuition. Judgment is hardly defined in a way that can be measured. This project therefore builds for hesitation, and measures it.

## Results

The milestones and the next task are in [docs/ROADMAP.md](docs/ROADMAP.md).

The benchmark runs end to end. A Rust harness sends probes to Ollama concurrently and writes a run log in JSONL that is validated against a schema. The Python side screens and scores the log.

**The first pilot could not measure Type 2 AUROC.** Every model answered all the answerable open questions correctly, so there were no wrong answers to separate from the right ones ([report](results/2026-08-17-M1-validity-screen.md)). Switching to forced choice made the metric measurable. A first report on forced choice over read its own numbers and was corrected after a second audit ([report](results/2026-08-17-M3b-audit-and-rerun.md)). The corrected item set was run twice per model.

Type 2 AUROC on answerable items only, with a cluster bootstrap over 32 independent items, for both runs:

| Model (q4_K_M) | Run 1 | Run 2 | Validity screen |
|---|---|---|---|
| `gemma4:12b` | **0.784 [0.552, 0.963]** | **0.815 [0.617, 0.964]** | 5 of 5 passed |
| `qwen3.5:9b` | 0.643 [0.419, 0.800] | 0.643 [0.419, 0.800] | Failed |
| `granite4.1:8b` | 0.520 [0.365, 0.678] | 0.544 [0.382, 0.713] | Failed |

**With 300 items the picture reversed** ([report](results/2026-08-17-doors-1-2.md)). Type 2 AUROC on 300 MMLU-Pro items, answerable items only, with cluster confidence intervals and two runs each:

| Model (q4_K_M) | Stated confidence | Agreement across 6 option orderings | Gate that requires unanimity |
|---|---|---|---|
| `granite4.1:8b` | 0.562 [0.514, 0.612] | **0.731 [0.674, 0.787]** | keeps 33% of items at 0.75 accuracy, against 0.43 without the gate |
| `gemma4:12b` | 0.530 [0.476, 0.582] | **0.737 [0.680, 0.794]** | keeps 49% at 0.79, against 0.59 |
| `qwen3.5:9b` | 0.514 [0.473, 0.555] | **0.746 [0.685, 0.806]** | keeps 36% at 0.85, against 0.55 |

Stated confidence is at chance level for all three models, while agreement across orderings reaches 0.73 to 0.75 for all three. The signal therefore depends on the method of elicitation, not on the model. The earlier result for `gemma4:12b` on 57 items was a small sample effect, and it is kept as the record of that.

On a question answering task over passages, a gate based on stated confidence does nothing. Checking that the answer appears in the passage, and requiring the same answer in 3 samples, reduces the wrong answers let through from 84 to 28.

The gate is a subcommand of the harness:

```bash
qualm-harness gate --model gemma4:12b --question "…" --passage "…"       # one question, routed
qualm-harness gate --model gemma4:12b --items datasets/squad2-task.jsonl  # gated against ungated
```

The command line also prints a pooled figure, which counts attempted answers to unanswerable questions as errors: 0.83, 0.64 and 0.65 to 0.67 for the three models. That figure measures how well a model separates answerable from unanswerable questions, not how well it anticipates its own errors, so it should not be quoted as the AUROC.

## Repository layout

| Path | Contents |
|---|---|
| `docs/RESEARCH.md` | The literature, checked at source, and what each finding means for the design |
| `docs/METRICS.md` | Why the project uses AUROC rather than the M ratio, and the validity screen |
| `docs/DESIGN.md` | The division between Rust and Python, and the reasons for it |
| `docs/ROADMAP.md` | Milestones, each with the result that would count as failure |
| `crates/qualm-harness/` | Rust: the control loop, concurrency and provenance |
| `python/qualm_analysis/` | Python: Type 2 signal detection, AUROC and validity screening |
| `schemas/` | The probe item and run record schemas. Provenance fields are required from version 0.3.0, and CI validates every committed log |
| `datasets/` | Probe items, versioned as data rather than code. See `datasets/README.md` |
| `results/` | Run logs and reports |

## Quick start

```bash
# Python analysis and tests
cd python && uv sync && uv run pytest -q

# Rust harness; rust-toolchain.toml pins the compiler version
cargo build --release
./target/release/qualm-harness bench --items datasets/fc-hard.jsonl \
  --model gemma4:12b --quant q4_K_M --permutations 2 --out results/raw/x.jsonl
```

The harness talks to Ollama at `http://localhost:11434` by default. Set `QUALM_OLLAMA` or pass `--ollama` to use another server. The measurements in `results/` were taken on an RTX 5060 Ti with 16 GB, with Ollama configured for 4 parallel requests (`OLLAMA_NUM_PARALLEL=4`).

## Where to start reading

Start with [docs/RESEARCH.md](docs/RESEARCH.md). Each design decision in this repository follows from a specific finding, and that document names which one.

## Licence

The code is released under the Apache License 2.0; see `LICENSE`. The datasets derived from public sources keep their own licences, which are listed in `datasets/README.md`.
