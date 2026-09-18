//! Probe items and mechanical scoring.
//!
//! Scoring lives here rather than on the Python side on purpose: a correctness
//! label must be decided identically in `bench` and `serve`, and duplicating the
//! rule across two languages is how the two silently diverge.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Item {
    pub id: String,
    pub prompt: String,
    pub answerable: bool,
    pub category: String,
    pub domain: String,
    #[serde(default)]
    pub difficulty: Option<String>,
    pub scoring: Scoring,
    /// Forced-choice alternatives in canonical order. Presence makes this a
    /// forced-choice item, which is what gives meta-d' a defined Type-1
    /// decision and gives AUROC a reachable negative class.
    #[serde(default)]
    pub options: Option<Vec<String>>,
    #[serde(default)]
    pub paraphrase_of: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    /// The passage the question is about, if any. The prompt already embeds
    /// it; this copy is what the gate's grounding check reads.
    #[serde(default)]
    pub context: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "method", rename_all = "kebab-case")]
pub enum Scoring {
    Exact {
        answer: String,
    },
    Numeric {
        answer: f64,
        tolerance: Option<f64>,
    },
    SetMatch {
        answer: Vec<String>,
    },
    /// Set-match where the ANSWER may contain the gold on token boundaries
    /// (one direction only — "Jani" must not match gold "Jani Beg"). For
    /// extractive QA (SQuAD-style) where the model answers "The Mongol army
    /// under Jani Beg" against gold "Jani Beg": exact match scores that
    /// confident correct answer as an error — the worst direction for Type-2
    /// AUROC. Over-credits a list answer that merely includes the gold; the
    /// tradeoff is stated in the schema.
    SetContains {
        answer: Vec<String>,
    },
    Regex {
        pattern: String,
    },
    /// Forced choice. `answer` is the correct option's TEXT, never its index —
    /// an index silently points at the wrong option once options are permuted,
    /// which is exactly what validity screen #4 does.
    Choice {
        answer: String,
    },
    /// Unanswerable by construction: any substantive answer is wrong.
    AlwaysAbstain,
}

/// Outcome of scoring one response against one item.
///
/// `correct` is deliberately an `Option`. A correct abstention on an
/// unanswerable item is a success but has no accuracy label — folding it into
/// `true` would inflate accuracy with items the model never attempted, and
/// folding it into `false` would punish exactly the behaviour we want.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub correct: Option<bool>,
    pub abstained: bool,
}

impl Item {
    /// Options as presented, given a permutation. Identity when `perm` is None.
    pub fn presented_options(&self, perm: Option<&[usize]>) -> Option<Vec<String>> {
        let opts = self.options.as_ref()?;
        Some(match perm {
            None => opts.clone(),
            Some(p) => p.iter().filter_map(|&i| opts.get(i).cloned()).collect(),
        })
    }

    /// Score a forced-choice reply. `choice` is a letter (A, B, ...) indexing
    /// into the options AS PRESENTED, so the permutation must be passed in —
    /// resolving against canonical order would score every permuted item wrong.
    pub fn score_choice(
        &self,
        abstain: bool,
        choice: Option<&str>,
        perm: Option<&[usize]>,
    ) -> Score {
        if abstain {
            return Score {
                correct: if self.answerable { Some(false) } else { None },
                abstained: true,
            };
        }
        let Scoring::Choice { answer } = &self.scoring else {
            // Not a forced-choice item; fall back so a mixed set still runs.
            return self.score(abstain, choice);
        };
        let Some(presented) = self.presented_options(perm) else {
            return Score {
                correct: Some(false),
                abstained: false,
            };
        };
        // Letter -> index. Anything else (empty, "AB", "1") is a miss, not a panic.
        let idx = choice
            .and_then(|c| c.trim().chars().next())
            .map(|ch| ch.to_ascii_uppercase())
            .filter(|ch| ch.is_ascii_uppercase())
            .map(|ch| (ch as u8 - b'A') as usize);
        let picked = idx.and_then(|i| presented.get(i));
        Score {
            correct: Some(picked.is_some_and(|p| norm(p) == norm(answer))),
            abstained: false,
        }
    }

    /// Apply a `premise_challenged` flag to an already-computed score.
    ///
    /// The prompt tells the model to pick an option AND set this flag if the
    /// question rests on something false — so a model can rebut a false
    /// premise while still returning a letter. Before 0.3.0 that reply was
    /// scored on the letter alone: on an unanswerable item it became a
    /// confident ERROR in the AUROC negative class, i.e. the model was
    /// punished for the exact behaviour the flag was added to make visible.
    /// (qwen: 4 records at confidence 0.95–1.0; granite: 2; the item
    /// `fc-fp-king-usa` alone accounted for most.)
    ///
    /// Rule: on an unanswerable item a premise rejection is a correct refusal
    /// — unscored, like a correct abstention (`correct: None`). On an
    /// answerable item it is a wrong call and stays an error, whatever the
    /// letter. Abstentions are unaffected. No answerable record in any run to
    /// date carried the flag, so the second branch is a decision, not a fit.
    pub fn with_premise_challenge(&self, score: Score, challenged: bool) -> Score {
        if !challenged || score.abstained {
            return score;
        }
        Score {
            correct: if self.answerable { Some(false) } else { None },
            abstained: false,
        }
    }

    pub fn score(&self, abstain: bool, answer: Option<&str>) -> Score {
        if abstain {
            return Score {
                // Abstaining on an answerable item is a miss; on an
                // unanswerable one it is the right call and unscored.
                correct: if self.answerable { Some(false) } else { None },
                abstained: true,
            };
        }

        let given = answer.unwrap_or("").trim();
        let correct = match &self.scoring {
            Scoring::AlwaysAbstain => false, // answered something unanswerable
            Scoring::Exact { answer } => norm(given) == norm(answer),
            Scoring::Numeric { answer, tolerance } => parse_number(given)
                .map(|v| (v - answer).abs() <= tolerance.unwrap_or(1e-9))
                .unwrap_or(false),
            Scoring::SetMatch { answer } => answer.iter().any(|a| norm(a) == norm(given)),
            Scoring::SetContains { answer } => {
                let g = norm(given);
                answer.iter().any(|a| contains_tokens(&g, &norm(a)))
            }
            Scoring::Regex { pattern } => match regex::Regex::new(pattern) {
                Ok(re) => re.is_match(given),
                // An uncompilable pattern would otherwise score EVERY response
                // wrong with no error recorded — a whole item reading as "the
                // model always failed". Loud is the only safe behaviour.
                Err(e) => panic!("item {}: uncompilable regex {pattern:?}: {e}", self.id),
            },
            // Reached only if a Choice item is scored through the open-ended
            // path, which score_choice() avoids.
            Scoring::Choice { .. } => false,
        };
        Score {
            correct: Some(correct),
            abstained: false,
        }
    }
}

/// `needle` appears in `hay` on token boundaries; both already normalised.
/// Empty needle never matches (an empty answer must not "contain" anything).
pub(crate) fn contains_tokens(hay: &str, needle: &str) -> bool {
    if needle.is_empty() || hay.is_empty() {
        return false;
    }
    let h: Vec<&str> = hay.split(' ').collect();
    let n: Vec<&str> = needle.split(' ').collect();
    n.len() <= h.len() && h.windows(n.len()).any(|w| w == n.as_slice())
}

/// Case- and punctuation-insensitive comparison.
///
/// Models pad short answers with articles and trailing periods ("the Paris.").
/// Without this, a correct answer scores wrong and the resulting AUROC measures
/// our string handling rather than the model's knowledge.
pub(crate) fn norm(s: &str) -> String {
    // Punctuation and whitespace are normalised FIRST. Doing it after
    // article-stripping made the stripping order-dependent: a quoted or
    // tab-separated answer ("\"The Nile\"") kept its article and mismatched.
    let cleaned: String = s
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    let collapsed = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    // Strip leading articles REPEATEDLY so normalisation is idempotent and both
    // sides converge. Stripping once was asymmetric: "the a team" -> "a team"
    // while the expected "a team" -> "team", so they failed to match.
    let mut s = collapsed.as_str();
    loop {
        let next = s
            .strip_prefix("the ")
            .or_else(|| s.strip_prefix("an "))
            .or_else(|| s.strip_prefix("a "));
        match next {
            Some(rest) => s = rest,
            None => break,
        }
    }
    s.to_string()
}

/// Spelled-out numerals, 0-20 plus the common round numbers.
///
/// Without this, `"Four"` and `"eight"` scored WRONG on items whose answers are
/// exactly those words — at confidence 1.0. The bias is one-directional (it can
/// only turn a correct confident answer into a confident error), which is the
/// worst possible direction for Type-2 AUROC, and it moved a published headline
/// by 0.07.
fn word_to_number(w: &str) -> Option<f64> {
    Some(match w {
        "zero" | "none" => 0.0,
        "one" => 1.0,
        "two" => 2.0,
        "three" => 3.0,
        "four" => 4.0,
        "five" => 5.0,
        "six" => 6.0,
        "seven" => 7.0,
        "eight" => 8.0,
        "nine" => 9.0,
        "ten" => 10.0,
        "eleven" => 11.0,
        "twelve" => 12.0,
        "thirteen" => 13.0,
        "fourteen" => 14.0,
        "fifteen" => 15.0,
        "sixteen" => 16.0,
        "seventeen" => 17.0,
        "eighteen" => 18.0,
        "nineteen" => 19.0,
        "twenty" => 20.0,
        "thirty" => 30.0,
        "forty" => 40.0,
        "fifty" => 50.0,
        "hundred" => 100.0,
        "thousand" => 1000.0,
        _ => return None,
    })
}

/// Pull the first number out of a short answer, tolerating units and separators.
///
/// Commas are STRIPPED, not turned into whitespace. Replacing them splits a
/// thousands-separated number so "299,792" parses as 299 — silently, and it
/// scores a correct answer as wrong. Caught by the tolerance test.
fn parse_number(s: &str) -> Option<f64> {
    let cleaned: String = s.chars().filter(|c| *c != ',' && *c != '_').collect();
    let digits = cleaned.split_whitespace().find_map(|tok| {
        tok.trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
            .parse::<f64>()
            .ok()
    });
    if let Some(n) = digits {
        // Scale words after a bare number: "1.5 million" was parsed as 1.5.
        let lower = cleaned.to_lowercase();
        let mult = if lower.contains("billion") {
            1e9
        } else if lower.contains("million") {
            1e6
        } else if lower.contains("thousand") {
            1e3
        } else {
            1.0
        };
        return Some(n * mult);
    }
    // No digits: try spelled-out numerals.
    cleaned
        .to_lowercase()
        .split(|c: char| !c.is_alphabetic())
        .find_map(word_to_number)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(answerable: bool, scoring: Scoring) -> Item {
        Item {
            id: "t".into(),
            prompt: "p".into(),
            answerable,
            category: "answerable".into(),
            domain: "d".into(),
            difficulty: None,
            scoring,
            options: None,
            paraphrase_of: None,
            source: None,
            context: None,
        }
    }

    fn choice_item(options: &[&str], correct: &str) -> Item {
        Item {
            id: "c".into(),
            prompt: "p".into(),
            answerable: true,
            category: "answerable".into(),
            domain: "d".into(),
            difficulty: None,
            scoring: Scoring::Choice {
                answer: correct.into(),
            },
            options: Some(options.iter().map(|s| s.to_string()).collect()),
            paraphrase_of: None,
            source: None,
            context: None,
        }
    }

    // ── regressions for the two verified scoring bugs ────────────────────

    #[test]
    fn spelled_numerals_score_correct() {
        // Both were logged at confidence 1.0 and scored WRONG, which is the
        // worst possible direction for Type-2 AUROC.
        let dna = item(
            true,
            Scoring::Numeric {
                answer: 4.0,
                tolerance: None,
            },
        );
        assert_eq!(dna.score(false, Some("Four")).correct, Some(true));
        let planets = item(
            true,
            Scoring::Numeric {
                answer: 8.0,
                tolerance: None,
            },
        );
        assert_eq!(planets.score(false, Some("eight")).correct, Some(true));
    }

    #[test]
    fn scale_words_multiply() {
        let it = item(
            true,
            Scoring::Numeric {
                answer: 1_500_000.0,
                tolerance: Some(1.0),
            },
        );
        assert_eq!(it.score(false, Some("1.5 million")).correct, Some(true));
    }

    #[test]
    fn norm_is_not_defeated_by_quotes_or_tabs() {
        // Article-stripping used to run before punctuation normalisation, so a
        // quoted or tab-separated answer kept its article and mismatched.
        let it = item(
            true,
            Scoring::Exact {
                answer: "Nile".into(),
            },
        );
        assert_eq!(it.score(false, Some("\"The Nile\"")).correct, Some(true));
        assert_eq!(it.score(false, Some("The\tNile")).correct, Some(true));
        assert_eq!(it.score(false, Some("the  nile.")).correct, Some(true));
    }

    #[test]
    fn article_stripping_is_idempotent_so_phrasings_converge() {
        // Stripping once was asymmetric: the answer side lost its article while
        // the response side kept one ("a team" -> "team" but "the a team" ->
        // "a team"), so equivalent phrasings failed to match. Greedy stripping
        // makes all of these normalise to the same string.
        for given in ["the a team", "a team", "team", "The A Team", "\"the team\""] {
            let it = item(
                true,
                Scoring::Exact {
                    answer: "team".into(),
                },
            );
            assert_eq!(
                it.score(false, Some(given)).correct,
                Some(true),
                "input {given:?}"
            );
        }
    }

    // ── forced choice ────────────────────────────────────────────────────

    #[test]
    fn choice_scores_against_canonical_order() {
        let it = choice_item(&["Paris", "Lyon", "Nice"], "Paris");
        assert_eq!(it.score_choice(false, Some("A"), None).correct, Some(true));
        assert_eq!(it.score_choice(false, Some("B"), None).correct, Some(false));
        assert_eq!(it.score_choice(false, Some("b"), None).correct, Some(false));
    }

    #[test]
    fn choice_resolves_against_the_presented_permutation() {
        // THE bug this design avoids: scoring a letter against canonical order
        // would mark every permuted item wrong. Rotate left by 1 =>
        // presented [Lyon, Nice, Paris], so the correct letter is now C.
        let it = choice_item(&["Paris", "Lyon", "Nice"], "Paris");
        let perm = vec![1, 2, 0];
        assert_eq!(
            it.score_choice(false, Some("C"), Some(&perm)).correct,
            Some(true)
        );
        assert_eq!(
            it.score_choice(false, Some("A"), Some(&perm)).correct,
            Some(false)
        );
    }

    #[test]
    fn choice_handles_garbage_letters_without_panicking() {
        let it = choice_item(&["Paris", "Lyon"], "Paris");
        for bad in ["", "Z", "1", "  ", "AB"] {
            let s = it.score_choice(false, Some(bad), None);
            assert!(!s.abstained);
            if bad == "AB" {
                // First char wins: 'A' is a legitimate read of "AB".
                assert_eq!(s.correct, Some(true));
            } else {
                assert_eq!(s.correct, Some(false), "input {bad:?}");
            }
        }
    }

    #[test]
    fn choice_abstention_follows_the_same_tri_state_rule() {
        let ans = choice_item(&["Paris", "Lyon"], "Paris");
        assert_eq!(ans.score_choice(true, None, None).correct, Some(false));
        let mut unans = choice_item(&["Paris", "Lyon"], "Paris");
        unans.answerable = false;
        assert_eq!(unans.score_choice(true, None, None).correct, None);
    }

    #[test]
    fn set_contains_matches_gold_inside_answer_on_token_boundaries() {
        let it = item(
            true,
            Scoring::SetContains {
                answer: vec!["Jani Beg".into(), "internal fertilization".into()],
            },
        );
        assert_eq!(
            it.score(false, Some("The Mongol army under Jani Beg"))
                .correct,
            Some(true)
        );
        assert_eq!(it.score(false, Some("Jani")).correct, Some(false)); // answer ⊂ gold: no
        assert_eq!(it.score(false, Some("fertilization")).correct, Some(false));
        assert_eq!(it.score(false, Some("Jani Begum")).correct, Some(false)); // token boundary
        assert_eq!(
            it.score(false, Some("internal fertilization and brood chambers"))
                .correct,
            Some(true)
        );
        assert_eq!(it.score(false, Some("")).correct, Some(false));
        assert_eq!(it.score(false, Some("Beg Jani")).correct, Some(false)); // order matters
    }

    #[test]
    fn premise_rejection_on_unanswerable_is_a_correct_refusal_not_an_error() {
        // The fc-fp-king-usa pattern: model picks a letter, flags the premise.
        let it = Item {
            answerable: false,
            ..choice_item(&["a", "b"], "a")
        };
        let base = it.score_choice(false, Some("B"), None);
        assert_eq!(base.correct, Some(false)); // scored on the letter alone
        let fixed = it.with_premise_challenge(base, true);
        assert_eq!(
            fixed,
            Score {
                correct: None,
                abstained: false
            }
        );
        // Flag absent: unchanged.
        assert_eq!(it.with_premise_challenge(base, false), base);
    }

    #[test]
    fn premise_rejection_on_answerable_is_an_error_even_with_the_right_letter() {
        let it = choice_item(&["a", "b"], "a");
        let base = it.score_choice(false, Some("A"), None);
        assert_eq!(base.correct, Some(true));
        assert_eq!(
            it.with_premise_challenge(base, true),
            Score {
                correct: Some(false),
                abstained: false
            }
        );
    }

    #[test]
    fn premise_flag_does_not_disturb_an_abstention() {
        let it = Item {
            answerable: false,
            ..choice_item(&["a", "b"], "a")
        };
        let base = it.score_choice(true, Some("A"), None);
        assert_eq!(it.with_premise_challenge(base, true), base);
    }

    #[test]
    fn presented_options_applies_the_permutation() {
        let it = choice_item(&["a", "b", "c"], "a");
        assert_eq!(it.presented_options(None).unwrap(), vec!["a", "b", "c"]);
        assert_eq!(
            it.presented_options(Some(&[2, 0, 1])).unwrap(),
            vec!["c", "a", "b"]
        );
    }

    #[test]
    fn exact_is_forgiving_about_articles_and_punctuation() {
        let it = item(
            true,
            Scoring::Exact {
                answer: "Paris".into(),
            },
        );
        assert_eq!(it.score(false, Some("Paris")).correct, Some(true));
        assert_eq!(it.score(false, Some("  paris. ")).correct, Some(true));
        assert_eq!(it.score(false, Some("The Paris")).correct, Some(true));
        assert_eq!(it.score(false, Some("Lyon")).correct, Some(false));
    }

    #[test]
    fn numeric_respects_tolerance_and_ignores_units() {
        let it = item(
            true,
            Scoring::Numeric {
                answer: 299792.0,
                tolerance: Some(1.0),
            },
        );
        assert_eq!(it.score(false, Some("299792 km/s")).correct, Some(true));
        assert_eq!(it.score(false, Some("299,792")).correct, Some(true));
        assert_eq!(it.score(false, Some("300000")).correct, Some(false));
    }

    #[test]
    fn abstaining_on_unanswerable_is_success_but_unscored() {
        let it = item(false, Scoring::AlwaysAbstain);
        let s = it.score(true, None);
        assert!(s.abstained);
        assert_eq!(s.correct, None, "must not inflate accuracy");
    }

    #[test]
    fn answering_an_unanswerable_item_is_wrong() {
        let it = item(false, Scoring::AlwaysAbstain);
        assert_eq!(it.score(false, Some("Atlantis City")).correct, Some(false));
    }

    #[test]
    fn abstaining_on_an_answerable_item_is_wrong() {
        let it = item(
            true,
            Scoring::Exact {
                answer: "Paris".into(),
            },
        );
        let s = it.score(true, None);
        assert!(s.abstained);
        assert_eq!(s.correct, Some(false), "over-abstention must be penalised");
    }

    #[test]
    fn set_match_accepts_any_alias() {
        let it = item(
            true,
            Scoring::SetMatch {
                answer: vec!["USA".into(), "United States".into()],
            },
        );
        assert_eq!(it.score(false, Some("united states")).correct, Some(true));
        assert_eq!(it.score(false, Some("Canada")).correct, Some(false));
    }
}
