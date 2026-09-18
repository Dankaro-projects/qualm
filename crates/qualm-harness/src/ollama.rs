//! Ollama transport.
//!
//! Small on purpose. The only subtle parts are documented below, and both cost
//! real debugging time on this stack already.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub struct Client {
    http: reqwest::Client,
    base: String,
}

#[derive(Debug, Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    /// Disabling model-side thinking is REQUIRED, not tuning. Reasoning models
    /// (qwen3.5) otherwise emit their JSON into `thinking` and leave `response`
    /// empty — measured as an apparent 0/12 against a model complying perfectly.
    think: bool,
    /// JSON Schema for grammar-constrained decoding. Must be the schema object,
    /// not the string "json": plain JSON mode constrains syntax but not field
    /// names, so the response parses and then fails to contain what we asked for.
    format: &'a Value,
    options: Options,
}

#[derive(Debug, Serialize, Clone)]
pub struct Options {
    pub temperature: f64,
    /// Generous by default. The schema constrains structure, not length, so a
    /// tight budget truncates mid-string and surfaces as `Unfinished string at
    /// EOF` — indistinguishable from the model being incapable.
    pub num_predict: i32,
    pub num_ctx: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            num_predict: 512,
            num_ctx: 8192,
            seed: Some(0),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct GenerateResponse {
    pub response: String,
    #[serde(default)]
    pub thinking: Option<String>,
    #[serde(default)]
    pub eval_count: Option<u32>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub done_reason: Option<String>,
}

impl Client {
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::builder()
                // Generous: a cold model load reads gigabytes from disk before
                // the first token, and a timeout here would look like a hang.
                .timeout(std::time::Duration::from_secs(600))
                .build()
                .expect("failed to build http client"),
            base: base.into(),
        }
    }

    pub async fn version(&self) -> Result<String> {
        let v: Value = self
            .http
            .get(format!("{}/api/version", self.base))
            .send()
            .await
            .context("Ollama unreachable; check that the server is running and QUALM_OLLAMA points at it")?
            .json()
            .await?;
        Ok(v["version"].as_str().unwrap_or("unknown").to_string())
    }

    pub async fn generate(
        &self,
        model: &str,
        prompt: &str,
        format: &Value,
        options: &Options,
    ) -> Result<GenerateResponse> {
        let req = GenerateRequest {
            model,
            prompt,
            stream: false,
            think: false,
            format,
            options: options.clone(),
        };
        let resp = self
            .http
            .post(format!("{}/api/generate", self.base))
            .json(&req)
            .send()
            .await
            .context("generate request failed")?;

        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            anyhow::bail!("Ollama returned {status}: {body}");
        }
        serde_json::from_str(&body).with_context(|| format!("unparseable response: {body}"))
    }
}

/// The model's structured reply. Mirrors schemas/probe-response.schema.json.
///
/// `deny_unknown_fields` and the range check in `validate()` exist because
/// `schema_valid: true` previously meant only "three fields deserialised". A
/// model returning `confidence: 42` was recorded as valid, and would have
/// inflated the variance screen into a PASS for the wrong reason.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeReply {
    pub abstain: bool,
    #[serde(default)]
    pub answer: Option<String>,
    pub confidence: f64,
    /// Forced-choice: the selected option letter, as presented.
    #[serde(default)]
    pub choice: Option<String>,
    /// The model believes the question rests on something false. Without this,
    /// a correct rebuttal is indistinguishable from credulity — which is
    /// exactly how gemma4's best answer got scored as its worst.
    #[serde(default)]
    pub premise_challenged: bool,
}

impl ProbeReply {
    /// Reject a reply the declared schema would have rejected.
    ///
    /// An incoherent reply must be flagged, never scored: `abstain:false` with
    /// no answer and no choice produced nothing, yet was recorded as a
    /// confident wrong answer. All four such records in the first run landed in
    /// the 13-item stratum carrying the headline.
    pub fn validate(&self, forced_choice: bool) -> Result<()> {
        if !(0.0..=1.0).contains(&self.confidence) || !self.confidence.is_finite() {
            anyhow::bail!("confidence {} outside [0,1]", self.confidence);
        }
        if !self.abstain {
            let has = if forced_choice {
                self.choice.as_deref().is_some_and(|c| !c.trim().is_empty())
            } else {
                self.answer.as_deref().is_some_and(|a| !a.trim().is_empty())
            };
            if !has {
                anyhow::bail!(
                    "abstain=false but no {} given",
                    if forced_choice { "choice" } else { "answer" }
                );
            }
        }
        Ok(())
    }
}

/// Parse the constrained reply out of a generate response.
///
/// Falls back to `thinking` when `response` is empty. That is not defensive
/// clutter: it is the exact failure that produced a wrong conclusion about
/// qwen3.5 before `think:false` was set, so the fallback both rescues the data
/// and records that it happened via the returned flag.
pub fn parse_reply(r: &GenerateResponse) -> Result<(ProbeReply, bool)> {
    let primary = r.response.trim();
    if !primary.is_empty() {
        if let Ok(p) = serde_json::from_str::<ProbeReply>(primary) {
            return Ok((p, false));
        }
    }
    if let Some(th) = r.thinking.as_deref().map(str::trim) {
        if !th.is_empty() {
            if let Ok(p) = serde_json::from_str::<ProbeReply>(th) {
                return Ok((p, true));
            }
        }
    }
    anyhow::bail!(
        "no parseable reply (done_reason={:?}, response={:?})",
        r.done_reason,
        truncate(primary, 160)
    )
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "…"
    }
}
