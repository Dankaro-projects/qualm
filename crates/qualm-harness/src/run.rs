//! The bench loop: read items, probe concurrently, write a run log.
//!
//! Single-pass for now — no retry, no aggregate. That is M1's requirement: the
//! validity screen needs one honest measurement per item, and adding the control
//! branches before knowing the signal is readable would be building on sand.
//! The FOK/JOL control loop is M4.

use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, Semaphore};

use crate::item::Item;
use crate::ollama::{parse_reply, Client, Options};

pub struct BenchConfig {
    /// Ollama server version, captured at run start.
    pub backend_version: String,
    /// How many option orderings to run each forced-choice item under.
    /// 1 = canonical only. >1 makes validity screen #4 (order stability)
    /// runnable — it never ran before because no permutations existed.
    pub permutations: usize,
    pub model: String,
    pub quantisation: String,
    pub run_id: String,
    pub concurrency: usize,
    pub options: Options,
    pub prompt_set: String,
}

/// The instruction wrapper around every item.
///
/// Two things here are load-bearing and were learned the hard way:
///
/// 1. **Abstention is stated to be unpenalised.** Without that, a model reads
///    the task as "answer or fail" and abstention rates measure perceived
///    pressure rather than uncertainty.
/// 2. **Brevity is demanded explicitly.** The schema constrains structure, not
///    length; models otherwise write paragraphs into `answer`, blow the token
///    budget, and the truncated JSON reads as incapability. Adding this took a
///    conformance rate from 6/12 to 16/16 on the same models.
///
/// Kept here rather than in the item file because it applies to every item;
/// `prompt_set` in the run log versions it, so a change to this text is visible
/// as a different set and does not silently pool with earlier results.
pub const PROMPT_TEMPLATE: &str = "\
Answer the question if you can. If the question cannot be answered — it is \
unanswerable, underspecified, rests on a false premise, or has no single correct \
answer — set abstain to true and answer to null. Declining is a valid choice and \
is not penalised; guessing when you cannot know is worse than declining.

Be terse: answer must be a short phrase, not a sentence. Give confidence as a \
number between 0 and 1.

Question: {question}";

/// Appended for forced-choice items.
///
/// Asks for a letter rather than the option text: text invites paraphrase, and
/// a paraphrased option cannot be matched back reliably. `premise_challenged`
/// is offered explicitly so rejecting the question is representable — otherwise
/// a model that spots a false premise has no move except to abstain or to play
/// along, and both get scored as if it had not spotted anything.
pub const CHOICE_SUFFIX: &str = "\n\nOptions:\n{options}\n\nReply with the letter of the \
correct option in `choice`. If the question itself rests on something false, set \
premise_challenged to true. If no option can be correct, set abstain to true.";

/// Rotate options left by `p`. Deterministic on purpose — an RNG here would
/// make a run irreproducible, and this harness is already non-deterministic
/// enough at the model layer. Rotation (not reversal) moves the correct answer
/// to a different position each time, which is what position bias needs.
fn permutation(len: usize, p: usize) -> Vec<usize> {
    (0..len).map(|i| (i + p) % len).collect()
}

pub async fn bench(
    client: Client,
    items: Vec<Item>,
    cfg: BenchConfig,
    out_path: &str,
) -> Result<Summary> {
    let schema: Value =
        serde_json::from_str(include_str!("../../../schemas/probe-response.schema.json"))
            .context("bundled probe-response schema is not valid JSON")?;
    // Ollama wants the bare schema; the $schema/$id/description keys are ours.
    //
    // TWO schemas, per item type. One shared schema does not work: with `answer`
    // required and `choice` optional, every model dutifully filled `answer` and
    // left `choice` empty, and 50 of 60 forced-choice probes were rejected. The
    // required list is what actually steers the model.
    let props = schema["properties"].clone();
    anyhow::ensure!(
        props.as_object().is_some_and(|o| !o.is_empty()),
        "probe-response schema has no usable `properties`"
    );
    let open_format = json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["abstain", "answer", "confidence"],
        "properties": props.clone(),
    });

    // Captured once and stamped on every record. The schema has always had the
    // field; leaving it null meant a run could not be attributed to a server
    // version — which became concrete when the inference server was upgraded mid-session
    // (kernel 6.17->7.0, driver 580.167->580.178) and prior logs became
    // unattributable to a stack state.
    let version = client.version().await?;
    let cfg = BenchConfig {
        backend_version: version.clone(),
        ..cfg
    };
    tracing::info!(ollama = %version, model = %cfg.model, items = items.len(), "starting bench");

    let file = tokio::fs::File::create(out_path)
        .await
        .with_context(|| format!("cannot create {out_path}"))?;
    let writer = Arc::new(Mutex::new(tokio::io::BufWriter::new(file)));

    // Bound in-flight work to the server's parallel slots. Above that, requests
    // queue inside Ollama and wall-clock gets worse, not better — measured.
    let sem = Arc::new(Semaphore::new(cfg.concurrency));
    let client = Arc::new(client);
    let cfg = Arc::new(cfg);
    let open_format = Arc::new(open_format);
    let props = Arc::new(props);

    let mut work: Vec<(Item, Option<Vec<usize>>)> = Vec::new();
    for item in items {
        match (&item.options, cfg.permutations) {
            (Some(o), k) if k > 1 && !o.is_empty() => {
                for p in 0..k.min(o.len()) {
                    work.push((item.clone(), Some(permutation(o.len(), p))));
                }
            }
            _ => work.push((item, None)),
        }
    }

    let mut tasks = Vec::with_capacity(work.len());
    for (item, perm) in work {
        let (sem, client, cfg, open_format, props, writer) = (
            sem.clone(),
            client.clone(),
            cfg.clone(),
            open_format.clone(),
            props.clone(),
            writer.clone(),
        );
        tasks.push(tokio::spawn(async move {
            // BEFORE acquire, so total_ms actually includes queue wait. It was
            // taken after, which made total_ms pure transport while the comment
            // and the run-record schema both claimed it captured queueing.
            let started = std::time::Instant::now();
            let _permit = sem.acquire().await.expect("semaphore closed");

            let mut prompt = PROMPT_TEMPLATE.replace("{question}", &item.prompt);
            let presented = item.presented_options(perm.as_deref());
            if let Some(opts) = &presented {
                let rendered = opts
                    .iter()
                    .enumerate()
                    .map(|(i, o)| format!("{}. {o}", (b'A' + i as u8) as char))
                    .collect::<Vec<_>>()
                    .join("\n");
                prompt.push_str(&CHOICE_SUFFIX.replace("{options}", &rendered));
            }
            let forced_choice = presented.is_some();

            // For forced choice, REQUIRE `choice` and constrain it to the exact
            // letters on offer. The enum is the point: grammar-constrained
            // decoding then makes an out-of-range or malformed letter
            // impossible, rather than something we have to score as wrong.
            let format = match &presented {
                None => (*open_format).clone(),
                Some(opts) => {
                    let letters: Vec<String> = (0..opts.len())
                        .map(|i| ((b'A' + i as u8) as char).to_string())
                        .collect();
                    let mut p = (*props).clone();
                    p["choice"] = json!({ "type": "string", "enum": letters });
                    json!({
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["abstain", "choice", "confidence"],
                        "properties": p,
                    })
                }
            };
            let result = client
                .generate(&cfg.model, &prompt, &format, &cfg.options)
                .await;

            let mut rec = base_record(&cfg, &item, perm.as_deref());
            let mut ok = false;

            match result {
                Err(e) => {
                    rec["error"] = json!(e.to_string());
                    rec["outcome"] = json!({
                        "answered": false, "correct": null,
                        "abstained": false, "schema_valid": false,
                        "control_action": "none", "attempts": 1
                    });
                }
                Ok(resp) => match parse_reply(&resp)
                    .and_then(|(r, ft)| r.validate(forced_choice).map(|_| (r, ft)))
                {
                    Err(e) => {
                        // A schema failure is a harness/format problem, NOT
                        // evidence about metacognition. Recorded separately so
                        // the two can never be conflated downstream.
                        rec["error"] = json!(e.to_string());
                        rec["outcome"] = json!({
                            "answered": false, "correct": null,
                            "abstained": false, "schema_valid": false,
                            "control_action": "none", "attempts": 1
                        });
                    }
                    Ok((reply, from_thinking)) => {
                        let score = if forced_choice {
                            item.score_choice(
                                reply.abstain,
                                reply.choice.as_deref(),
                                perm.as_deref(),
                            )
                        } else {
                            item.score(reply.abstain, reply.answer.as_deref())
                        };
                        let score = item.with_premise_challenge(score, reply.premise_challenged);
                        ok = true;
                        // Its own field, not `error`. `error` is the hard-failure
                        // channel the CLI points readers at; putting a benign
                        // warning there made `error != null` ambiguous.
                        rec["recovered_from_thinking"] = json!(from_thinking);
                        rec["signals"] = json!({
                            "confidence": reply.confidence,
                            "fok": null, "jol": null, "agreement": null
                        });
                        rec["outcome"] = json!({
                            "answered": !reply.abstain,
                            "correct": score.correct,
                            "raw_answer": reply.answer,
                            "choice": reply.choice,
                            "premise_challenged": reply.premise_challenged,
                            "abstained": score.abstained,
                            "schema_valid": true,
                            "control_action": "none",
                            "attempts": 1
                        });
                        rec["timing"] = json!({
                            // Ours includes queue wait behind the semaphore;
                            // Ollama's excludes it. Both are recorded because
                            // the difference IS the queueing cost, and having
                            // only one of them makes that unrecoverable.
                            "total_ms": started.elapsed().as_millis() as u64,
                            "ollama_total_ns": resp.total_duration,
                            "eval_tokens": resp.eval_count,
                            "prompt_tokens": resp.prompt_eval_count
                        });
                    }
                },
            }

            let line = format!(
                "{}\n",
                serde_json::to_string(&rec).expect("record serialises")
            );
            let mut w = writer.lock().await;
            w.write_all(line.as_bytes())
                .await
                .expect("run log write failed");
            ok
        }));
    }

    let mut summary = Summary::default();
    for t in tasks {
        match t.await {
            Ok(true) => summary.ok += 1,
            Ok(false) => summary.failed += 1,
            Err(e) => {
                tracing::error!(error = %e, "probe task panicked");
                summary.failed += 1;
            }
        }
    }

    // Explicit flush: BufWriter drops silently, and a truncated run log that
    // looks complete is the worst possible artifact here.
    let mut w = writer.lock().await;
    w.flush().await.context("failed flushing run log")?;
    Ok(summary)
}

fn base_record(cfg: &BenchConfig, item: &Item, perm: Option<&[usize]>) -> Value {
    json!({
        "run_id": cfg.run_id,
        "item_id": item.id,
        "ts": now_rfc3339(),
        // Item metadata is COPIED into the record rather than joined on item_id
        // at analysis time. The run log is the archival artifact — a reviewer
        // must be able to stratify by category and domain from the log alone,
        // years later, without also needing the exact item file that produced
        // it. Duplication is the cheaper failure than a broken join.
        "item": {
            "answerable": item.answerable,
            "forced_choice": item.options.is_some(),
            "category": item.category,
            "domain": item.domain,
            "difficulty": item.difficulty,
            "paraphrase_of": item.paraphrase_of,
            "source": item.source,
        },
        "model": {
            "name": cfg.model,
            // Required, and the reason is in docs/METRICS.md: calibration is not
            // comparable across quant levels, so an untagged row is unscoreable.
            "quantisation": cfg.quantisation,
            "backend": "ollama",
            "backend_version": cfg.backend_version,
        },
        "sampling": {
            "temperature": cfg.options.temperature,
            "num_ctx": cfg.options.num_ctx,
            "num_predict": cfg.options.num_predict,
            "num_parallel": cfg.concurrency,
            "seed": cfg.options.seed,
            "think": false,
        },
        "elicitation": { "strategy": "verbalised" },
        // Bumped whenever the RECORD SHAPE changes, not only on releases. It
        // stayed at 0.1.0 through an earlier shape change, so a stale log was
        // indistinguishable from a current one.
        "harness_version": env!("CARGO_PKG_VERSION"),
        "prompt_set": cfg.prompt_set,
        "error": null,
        "permutation": perm,
    })
}

fn now_rfc3339() -> String {
    time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "unknown".into())
}

#[derive(Debug, Default)]
pub struct Summary {
    pub ok: usize,
    pub failed: usize,
}

pub fn load_items(path: &str) -> Result<Vec<Item>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {path}"))?;
    let mut items = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        items.push(
            serde_json::from_str::<Item>(line)
                .with_context(|| format!("{path}:{}: bad item", n + 1))?,
        );
    }
    anyhow::ensure!(!items.is_empty(), "{path} contained no items");
    Ok(items)
}
