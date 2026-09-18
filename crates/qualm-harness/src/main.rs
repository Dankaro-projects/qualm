//! qualm-harness — elicit a confidence signal, then act on it.
//!
//!   bench  — probe every item once, record everything, act on nothing (M1–M3)
//!   gate   — one question (or a batch with ground truth), routed
//!            accept / review / decline by grounding and agreement (M4)
//!
//! They share the client, prompt and response schema, because a benchmark that
//! does not exercise the production path measures something other than
//! production — which is the failure mode this whole repo is about.

mod gate;
mod item;
mod ollama;
mod run;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "qualm",
    version,
    about = "Metacognitive harness for local LLMs"
)]
struct Cli {
    /// Ollama endpoint. Set QUALM_OLLAMA or pass --ollama when the server
    /// runs on another machine.
    #[arg(long, env = "QUALM_OLLAMA", default_value = "http://localhost:11434")]
    ollama: String,

    /// Bound on in-flight requests. Ollama here runs OLLAMA_NUM_PARALLEL=4;
    /// above that requests queue inside the server and wall-clock gets worse,
    /// so the default matches the server rather than guessing.
    #[arg(long, default_value_t = 4)]
    concurrency: usize,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run probes and write a run log. Records; does not gate.
    Bench {
        /// Items file (JSONL, per schemas/probe-item.schema.json)
        #[arg(long)]
        items: String,
        /// Model tag as Ollama knows it, e.g. granite4.1:8b
        #[arg(long)]
        model: String,
        /// Quantisation tag. Required: calibration numbers are not comparable
        /// across quant levels, so an untagged run cannot be scored at all.
        #[arg(long)]
        quant: String,
        #[arg(long, default_value = "results/run.jsonl")]
        out: String,
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long, default_value_t = 512)]
        num_predict: i32,
        #[arg(long, default_value_t = 8192)]
        num_ctx: i32,
        /// Identifier for the prompt wording. Change it when PROMPT_TEMPLATE
        /// changes, so old and new results cannot silently pool.
        #[arg(long, default_value = "v2")]
        prompt_set: String,
        /// Option orderings per forced-choice item. >1 enables validity screen
        /// #4 (order stability), which could not run at all before.
        #[arg(long, default_value_t = 1)]
        permutations: usize,
        /// Sampling seed. Vary it with --temperature > 0 to run the same items
        /// several times for self-consistency (agreement across seeded runs);
        /// leave at 0 for the deterministic measurement.
        #[arg(long, default_value_t = 0)]
        seed: i64,
    },

    /// Check the endpoint answers and report its version.
    Ping,

    /// Route one question — or every item in a file — as accept / review /
    /// decline. Grounding first (answer must appear in the passage), then
    /// agreement across N samples; stated confidence is a tie-breaker only.
    /// With --items and ground truth it prints gated-vs-ungated.
    Gate {
        #[arg(long)]
        model: String,
        /// One question. Mutually exclusive with --items.
        #[arg(long, conflicts_with = "items")]
        question: Option<String>,
        /// Passage for --question; enables the grounding check.
        #[arg(long, requires = "question")]
        passage: Option<String>,
        /// Items file (JSONL); items with `context` get the grounding check.
        #[arg(long)]
        items: Option<String>,
        /// Per-item decisions, JSONL. Only with --items.
        #[arg(long)]
        out: Option<String>,
        /// Deterministic call plus N-1 seeded samples. 1 = no agreement gate.
        #[arg(long, default_value_t = 3)]
        samples: usize,
        #[arg(long, default_value_t = 0.7)]
        sample_temperature: f64,
        /// Route to review unless the answer appears verbatim in the passage.
        /// Ignored when there is no passage.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        require_grounded: bool,
        /// Stated-confidence tie-breaker. At n=300 confidence is at chance on
        /// this stack, so this is not the gate; kept for the table.
        #[arg(long, default_value_t = 1.0)]
        accept_at: f64,
        #[arg(long, default_value_t = 512)]
        num_predict: i32,
        #[arg(long, default_value_t = 8192)]
        num_ctx: i32,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "qualm_harness=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let client = ollama::Client::new(&cli.ollama);

    match cli.command {
        Command::Ping => {
            println!("{} — ollama {}", cli.ollama, client.version().await?);
            Ok(())
        }

        Command::Bench {
            items,
            model,
            quant,
            out,
            temperature,
            num_predict,
            num_ctx,
            prompt_set,
            permutations,
            seed,
        } => {
            let items = run::load_items(&items)?;
            let cfg = run::BenchConfig {
                // Overwritten in bench() from the live server; this is a
                // placeholder so the struct is constructible before we connect.
                backend_version: String::new(),
                model: model.clone(),
                quantisation: quant,
                // Derived from the wall clock rather than random, so a run is
                // identifiable without needing an RNG in the provenance.
                run_id: format!(
                    "{}-{}",
                    model.replace([':', '.', '/'], "-"),
                    time::OffsetDateTime::now_utc().unix_timestamp()
                ),
                concurrency: cli.concurrency,
                options: ollama::Options {
                    temperature,
                    num_predict,
                    num_ctx,
                    seed: Some(seed),
                },
                prompt_set,
                permutations,
            };
            let n_items = items.len();
            let summary = run::bench(client, items, cfg, &out).await?;
            // Total is ok+failed, not item count: with permutations there are
            // several probes per item, and "116/58" read like a bug.
            println!(
                "{}/{} probes recorded ({} failed) over {n_items} items -> {out}",
                summary.ok,
                summary.ok + summary.failed,
                summary.failed
            );
            if summary.failed > 0 {
                // Loud but not fatal: a partial run is still analysable, and
                // silently exiting 0 on a half-empty log would be worse.
                eprintln!(
                    "warning: {} probe(s) produced no usable reply; \
                     check `error` and `schema_valid` in the log before scoring",
                    summary.failed
                );
            }
            Ok(())
        }

        Command::Gate {
            model,
            question,
            passage,
            items,
            out,
            samples,
            sample_temperature,
            require_grounded,
            accept_at,
            num_predict,
            num_ctx,
        } => {
            let cfg = gate::GateConfig {
                model,
                samples,
                sample_temperature,
                require_grounded,
                accept_at,
                concurrency: cli.concurrency,
                options: ollama::Options {
                    temperature: 0.0,
                    num_predict,
                    num_ctx,
                    seed: Some(0),
                },
            };
            match (question, items) {
                (Some(q), None) => {
                    let out = gate::gate_one(&client, &cfg, &q, passage.as_deref()).await;
                    println!("{}", serde_json::to_string_pretty(&out)?);
                    Ok(())
                }
                (None, Some(path)) => {
                    let items = run::load_items(&path)?;
                    let n = items.len();
                    let summary =
                        gate::gate_items(client, cfg.clone(), items, out.as_deref()).await?;
                    print!("{}", summary.render(&cfg));
                    if let Some(p) = out {
                        println!("{n} decisions -> {p}");
                    }
                    Ok(())
                }
                _ => anyhow::bail!("gate needs exactly one of --question or --items"),
            }
        }
    }
}
