//! The gate: one model call, routed — `accept` / `review` / `decline`.
//!
//! What this checks, in order, is what Door 2 measured
//! (results/2026-08-17-doors-1-2.md §2, gemma4 on SQuAD2 passage-QA):
//!
//! 1. **Grounding** — for a passage task, the answer must appear verbatim in
//!    the passage (after `norm`). Costs nothing; alone it took errors let
//!    through from 84 to 47 with a review pile that was 90% real errors.
//! 2. **Agreement** — ask the item N times (deterministic call plus N−1 seeded
//!    samples at temperature > 0) and accept only if every sample says the same
//!    thing. 84 → 37 alone; grounded AND unanimous 84 → 28.
//! 3. **Stated confidence** — never the gate. At n=300 it is at chance for
//!    every model on this stack; `accept_at` is a tie-breaker only.
//!
//! The ceiling is consistent hallucination — an unanswerable question answered
//! the same way every time — which no self-referential check can see. That is
//! why this reports the errors it *let through*, not only the ones it caught.
//!
//! The prompt, response schema and `norm()` are the harness's own, so what
//! runs here is what the benchmark measured.

use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

use crate::item::{contains_tokens, norm, Item, Scoring};
use crate::ollama::{parse_reply, Client, Options};
use crate::run::PROMPT_TEMPLATE;

/// Same wrapper `scripts/convert-squad2.py` uses, so a `--passage` given on
/// the command line renders exactly like a dataset item.
pub const PASSAGE_TEMPLATE: &str =
    "Using ONLY the passage below, answer the question. If the passage \
does not contain the answer, decline.\n\nPassage: {ctx}\n\nQuestion: {q}";

#[derive(Debug, Clone)]
pub struct GateConfig {
    pub model: String,
    pub samples: usize,
    pub sample_temperature: f64,
    pub require_grounded: bool,
    pub accept_at: f64,
    pub concurrency: usize,
    pub options: Options,
}

#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    pub seed: i64,
    pub temperature: f64,
    pub abstain: bool,
    pub answer: Option<String>,
    pub confidence: f64,
    pub premise_challenged: bool,
    /// The reply parsed but was not a usable answer or abstention
    /// (`abstain:false` with no answer, out-of-range confidence, transport
    /// error). Never accepted; routed to review.
    pub malformed: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Decision {
    Accept,
    Review,
    Decline,
}

#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub answer: Option<String>,
    /// Mean stated confidence over the samples that agree with the majority.
    pub confidence: f64,
    pub abstained: bool,
    pub premise_challenged: bool,
    /// Share of samples agreeing with the majority (1.0 = unanimous).
    pub agreement: f64,
    /// `None` when there is no passage or no answer to ground.
    pub grounded: Option<bool>,
    pub malformed: bool,
    pub decision: Decision,
    pub samples: Vec<Sample>,
}

fn open_format() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["abstain", "answer", "confidence"],
        "properties": {
            "abstain": {"type": "boolean"},
            "answer": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "premise_challenged": {"type": "boolean"}
        }
    })
}

async fn one_sample(
    client: &Client,
    cfg: &GateConfig,
    prompt: &str,
    seed: i64,
    temperature: f64,
    format: &Value,
) -> Sample {
    let opts = Options {
        temperature,
        seed: Some(seed),
        ..cfg.options.clone()
    };
    let mut s = Sample {
        seed,
        temperature,
        abstain: false,
        answer: None,
        confidence: 0.0,
        premise_challenged: false,
        malformed: true,
        error: None,
    };
    match client.generate(&cfg.model, prompt, format, &opts).await {
        Err(e) => s.error = Some(e.to_string()),
        Ok(resp) => match parse_reply(&resp) {
            Err(e) => s.error = Some(e.to_string()),
            Ok((r, _)) => {
                s.abstain = r.abstain;
                s.answer = r.answer.clone();
                s.confidence = r.confidence;
                s.premise_challenged = r.premise_challenged;
                // Usable iff it validates as an open-ended reply. The failure
                // that matters is `abstain:false` with no answer — gemma4 does
                // it on ~10% of passage-QA items, all unanswerable, a decline
                // expressed wrongly. Recorded, never accepted.
                s.malformed = r.validate(false).is_err();
            }
        },
    }
    s
}

/// Run the gate on one rendered prompt. `passage` is only used for grounding.
pub async fn gate_prompt(
    client: &Client,
    cfg: &GateConfig,
    prompt: &str,
    passage: Option<&str>,
    sem: &Semaphore,
) -> Outcome {
    let format = open_format();
    let mut samples = Vec::with_capacity(cfg.samples.max(1));
    // Sample 0 is the deterministic call (temp 0, seed 0) — the thing that was
    // benchmarked. The rest vary the seed at sample_temperature so agreement
    // measures the model's own consistency, not the decoder's.
    let plan: Vec<(i64, f64)> = std::iter::once((0, 0.0))
        .chain((1..cfg.samples.max(1) as i64).map(|s| (s, cfg.sample_temperature)))
        .collect();
    for (seed, temp) in plan {
        let _permit = sem.acquire().await.expect("semaphore closed");
        samples.push(one_sample(client, cfg, prompt, seed, temp, &format).await);
    }

    let malformed = samples.iter().any(|s| s.malformed);
    // Vote key: DECLINE for an abstention or a malformed reply, else the
    // normalised answer. Same rule as the analysis side.
    let keys: Vec<String> = samples
        .iter()
        .map(|s| {
            if s.abstain || s.malformed {
                "\u{0}DECLINE".to_string()
            } else {
                norm(s.answer.as_deref().unwrap_or(""))
            }
        })
        .collect();
    let mut counts: std::collections::HashMap<&str, usize> = Default::default();
    for k in &keys {
        *counts.entry(k.as_str()).or_default() += 1;
    }
    let (maj, n_maj) = counts
        .iter()
        .max_by_key(|(k, n)| (**n, std::cmp::Reverse((*k).to_string())))
        .map(|(k, n)| (k.to_string(), *n))
        .unwrap_or_default();
    let agreement = n_maj as f64 / samples.len().max(1) as f64;
    let majority_declined = maj == "\u{0}DECLINE";
    let agreeing: Vec<&Sample> = samples
        .iter()
        .zip(&keys)
        .filter(|(_, k)| **k == maj)
        .map(|(s, _)| s)
        .collect();
    let answer = if majority_declined {
        None
    } else {
        agreeing.first().and_then(|s| s.answer.clone())
    };
    let confidence = if agreeing.is_empty() {
        0.0
    } else {
        agreeing.iter().map(|s| s.confidence).sum::<f64>() / agreeing.len() as f64
    };

    let grounded = match (passage, &answer) {
        (Some(p), Some(a)) => {
            let a = norm(a);
            Some(!a.is_empty() && contains_tokens(&norm(p), &a))
        }
        _ => None,
    };

    let decision = if malformed && !majority_declined {
        Decision::Review
    } else if majority_declined {
        if malformed {
            Decision::Review
        } else {
            Decision::Decline
        }
    } else if cfg.require_grounded && grounded == Some(false) {
        Decision::Review
    } else if confidence >= cfg.accept_at && agreement >= 1.0 {
        Decision::Accept
    } else {
        Decision::Review
    };

    Outcome {
        answer,
        confidence,
        abstained: majority_declined,
        premise_challenged: samples.first().is_some_and(|s| s.premise_challenged),
        agreement,
        grounded,
        malformed,
        decision,
        samples,
    }
}

/// Single question from the command line.
pub async fn gate_one(
    client: &Client,
    cfg: &GateConfig,
    question: &str,
    passage: Option<&str>,
) -> Outcome {
    let q = match passage {
        Some(p) => PASSAGE_TEMPLATE
            .replace("{ctx}", p)
            .replace("{q}", question),
        None => question.to_string(),
    };
    let prompt = PROMPT_TEMPLATE.replace("{question}", &q);
    let sem = Semaphore::new(cfg.concurrency.max(1));
    gate_prompt(client, cfg, &prompt, passage, &sem).await
}

// ── batch mode: gated vs ungated on items with ground truth ─────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Row {
    pub item_id: String,
    pub answerable: bool,
    pub category: String,
    /// Task-correct for the base (deterministic) output taken as-is: a
    /// matching answer, or a decline on an unanswerable item.
    pub task_correct: bool,
    pub decision: Decision,
    pub confidence: f64,
    pub agreement: f64,
    pub grounded: Option<bool>,
    pub malformed: bool,
    pub answer: Option<String>,
}

fn task_correct(item: &Item, out: &Outcome) -> bool {
    match &out.answer {
        None => !item.answerable,
        Some(a) => {
            if matches!(item.scoring, Scoring::AlwaysAbstain) {
                false
            } else {
                item.score(false, Some(a)).correct.unwrap_or(false)
            }
        }
    }
}

pub struct GateSummary {
    pub rows: Vec<Row>,
}

impl GateSummary {
    fn line(&self, name: &str, accept: impl Fn(&Row) -> bool) -> String {
        let answered: Vec<&Row> = self.rows.iter().filter(|r| r.answer.is_some()).collect();
        let n = self.rows.len().max(1);
        let acc: Vec<&&Row> = answered.iter().filter(|r| accept(r)).collect();
        let rev: Vec<&&Row> = answered.iter().filter(|r| !accept(r)).collect();
        let acc_correct = acc.iter().filter(|r| r.task_correct).count();
        let acc_err = acc.len() - acc_correct;
        let rev_err = rev.iter().filter(|r| !r.task_correct).count();
        format!(
            "  {name:34} accept {:>3.0}% · accepted accuracy {:.3} · errors let through {acc_err:>3} · to review {:>3} ({rev_err} real errors)",
            100.0 * acc.len() as f64 / n as f64,
            if acc.is_empty() { f64::NAN } else { acc_correct as f64 / acc.len() as f64 },
            rev.len(),
        )
    }

    pub fn render(&self, cfg: &GateConfig) -> String {
        let n = self.rows.len();
        let ung = self.rows.iter().filter(|r| r.task_correct).count();
        let answered = self.rows.iter().filter(|r| r.answer.is_some()).count();
        let ans_err = self
            .rows
            .iter()
            .filter(|r| r.answer.is_some() && !r.task_correct)
            .count();
        let halluc = self
            .rows
            .iter()
            .filter(|r| r.answer.is_some() && !r.answerable)
            .count();
        let has_ctx = self.rows.iter().any(|r| r.grounded.is_some());
        let mut s = String::new();
        s.push_str(&format!(
            "GATE  {} · samples {} @ {:.2} · require_grounded {} · accept_at {}\n",
            cfg.model, cfg.samples, cfg.sample_temperature, cfg.require_grounded, cfg.accept_at
        ));
        s.push_str(&format!(
            "n={n} · ungated task accuracy {:.3} (errors {}) · answered {answered} (of which {ans_err} wrong: {halluc} answered-unanswerable)\n",
            ung as f64 / n.max(1) as f64,
            n - ung
        ));
        s.push_str("\nGATES on answered items  (declined items pass through as declines)\n");
        s.push_str(&self.line("none", |_| true));
        s.push('\n');
        s.push_str(
            &self.line(&format!("stated confidence >= {}", cfg.accept_at), |r| {
                r.confidence >= cfg.accept_at
            }),
        );
        s.push('\n');
        if has_ctx {
            s.push_str(&self.line("grounded (answer in passage)", |r| r.grounded == Some(true)));
            s.push('\n');
        }
        s.push_str(&self.line("unanimous across samples", |r| r.agreement >= 1.0));
        s.push('\n');
        if has_ctx {
            s.push_str(&self.line("grounded AND unanimous", |r| {
                r.grounded == Some(true) && r.agreement >= 1.0
            }));
            s.push('\n');
        }
        s.push_str(&self.line("== configured decision ==", |r| {
            r.decision == Decision::Accept
        }));
        s.push('\n');
        s
    }
}

pub async fn gate_items(
    client: Client,
    cfg: GateConfig,
    items: Vec<Item>,
    out_path: Option<&str>,
) -> Result<GateSummary> {
    let client = Arc::new(client);
    let cfg = Arc::new(cfg);
    let sem = Arc::new(Semaphore::new(cfg.concurrency.max(1)));
    let mut tasks = Vec::with_capacity(items.len());
    for item in items {
        let (client, cfg, sem) = (client.clone(), cfg.clone(), sem.clone());
        tasks.push(tokio::spawn(async move {
            let prompt = PROMPT_TEMPLATE.replace("{question}", &item.prompt);
            let out = gate_prompt(&client, &cfg, &prompt, item.context.as_deref(), &sem).await;
            let row = Row {
                item_id: item.id.clone(),
                answerable: item.answerable,
                category: item.category.clone(),
                task_correct: task_correct(&item, &out),
                decision: out.decision,
                confidence: out.confidence,
                agreement: out.agreement,
                grounded: out.grounded,
                malformed: out.malformed,
                answer: out.answer.clone(),
            };
            (row, out)
        }));
    }
    let mut rows = Vec::new();
    let mut lines = Vec::new();
    for t in tasks {
        let (row, out) = t.await.context("gate task panicked")?;
        lines.push(serde_json::to_string(&json!({
            "item_id": row.item_id, "answerable": row.answerable, "category": row.category,
            "task_correct": row.task_correct, "outcome": out,
        }))?);
        rows.push(row);
    }
    if let Some(p) = out_path {
        let mut f = tokio::fs::File::create(p)
            .await
            .with_context(|| format!("cannot create {p}"))?;
        for l in &lines {
            f.write_all(l.as_bytes()).await?;
            f.write_all(b"\n").await?;
        }
        f.flush().await?;
    }
    Ok(GateSummary { rows })
}
