#!/usr/bin/env -S uv run --quiet python
"""Generate datasets/fc-hard.jsonl — hard forced-choice items.

WHY THIS EXISTS. The open-ended pilot could not measure Type-2 AUROC at all:
all three models answered all 46 answerable items correctly, so the negative
class was empty and AUROC was undefined rather than weak. Separately, meta-d'
and M-ratio require a two-alternative Type-1 decision that open-ended QA does
not have (arXiv 2603.25112).

Forced choice fixes both, but only if the items are genuinely hard. THE DESIGN
GOAL IS AN ERROR RATE, not a topic list: target 15-25% wrong. An item every
model aces contributes nothing to a discrimination metric.

How difficulty is manufactured here, since "obscure" is not the same as "hard":
  1. Near-miss distractors — the plausible wrong answer people actually give
     (Everest's height off by a plausible amount; the runner-up city).
  2. Commonly-misremembered facts, where the popular answer is a distractor.
  3. Fine discriminations — dates within a few years, adjacent categories.
  4. Superlatives with a contested runner-up.

Options are stored in CANONICAL order. The harness permutes them at run time
(`--permutations`), which is what finally makes validity screen #4 runnable.
"""

import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "datasets" / "fc-hard.jsonl"
items: list[dict] = []


def fc(iid, prompt, options, correct, domain, difficulty="hard", twin=None):
    assert correct in options, f"{iid}: correct answer not among options"
    assert len(options) == len(set(options)), f"{iid}: duplicate options"
    items.append({
        "id": iid,
        "prompt": prompt,
        "answerable": True,
        "category": "answerable",
        "domain": domain,
        "difficulty": difficulty,
        "options": options,
        "scoring": {"method": "choice", "answer": correct},
        **({"paraphrase_of": twin} if twin else {}),
        "source": "authored",
    })


def unanswerable(iid, prompt, options, category, domain, twin=None):
    """Unanswerable, but still presented with options.

    The options matter: an unanswerable item WITHOUT plausible options is
    trivially detectable, which is how the pilot ended up with a 100% ceiling on
    obvious false premises. Every option here is superficially reasonable, so
    declining requires noticing the problem rather than noticing that nothing
    fits.
    """
    items.append({
        "id": iid,
        "prompt": prompt,
        "answerable": False,
        "category": category,
        "domain": domain,
        "options": options,
        "scoring": {"method": "always-abstain"},
        **({"paraphrase_of": twin} if twin else {}),
        "source": "authored",
    })


# ── HARD ANSWERABLE — near-miss and commonly-misremembered ──────────────────

fc("fc-everest-height", "What is the height of Mount Everest above sea level, to the nearest metre?",
   ["8848 m", "8849 m", "8850 m", "8844 m"], "8849 m", "geography")
# REMOVED fc-nile-vs-amazon: Nile-vs-Amazon length is genuinely disputed, so the
# ground truth is contested and a "wrong" answer is not an error. Missed by 2 of 3
# models on the first run, which is the tell.
fc("fc-largest-lake", "What is the largest lake in the world by surface area?",
   ["Caspian Sea", "Lake Superior", "Lake Victoria", "Lake Baikal"], "Caspian Sea", "geography", "medium")
fc("fc-deepest-lake", "Which lake is the deepest in the world?",
   ["Lake Baikal", "Caspian Sea", "Lake Tanganyika", "Crater Lake"], "Lake Baikal", "geography", "medium")
fc("fc-southernmost-capital", "Which is the southernmost national capital city in the world?",
   ["Wellington", "Canberra", "Santiago", "Buenos Aires"], "Wellington", "geography")
# REMOVED fc-most-borders: China and Russia BOTH border 14 countries, so the item
# has no unique answer. Missed by all three models — an item every model fails is
# a defective item far more often than it is a shared blind spot.
fc("fc-turkey-continent-area", "Which continent contains the majority of Turkey's land area?",
   ["Asia", "Europe", "Africa", "split exactly evenly"], "Asia", "geography", "medium")
fc("fc-largest-island", "What is the largest island in the world?",
   ["Greenland", "Australia", "New Guinea", "Borneo"], "Greenland", "geography", "medium")

fc("fc-lightning-temp", "Roughly how hot is a lightning bolt's channel at peak?",
   ["about 30,000 K", "about 5,000 K", "about 1,000 K", "about 300,000 K"], "about 30,000 K", "science")
fc("fc-human-bones-adult", "How many bones are in the adult human body?",
   ["206", "212", "198", "220"], "206", "science", "medium")
fc("fc-human-bones-baby", "Approximately how many bones is a human born with?",
   ["about 300", "about 206", "about 150", "about 400"], "about 300", "science")
fc("fc-blood-colour", "What colour is deoxygenated human blood in the veins?",
   ["dark red", "blue", "purple", "colourless"], "dark red", "science", "medium")
fc("fc-tongue-taste-map", "Which statement about taste perception on the tongue is correct?",
   ["All taste regions detect all basic tastes", "Sweet is detected only at the tip",
    "Bitter is detected only at the sides", "Sour is detected only at the back"],
   "All taste regions detect all basic tastes", "science")
fc("fc-gravity-moon-fraction", "Surface gravity on the Moon is approximately what fraction of Earth's?",
   ["1/6", "1/3", "1/10", "1/2"], "1/6", "science", "medium")
fc("fc-speed-sound", "What is the approximate speed of sound in dry air at 20 degrees Celsius?",
   ["343 m/s", "300 m/s", "331 m/s", "365 m/s"], "343 m/s", "science")
fc("fc-heaviest-noble-gas", "Which of these noble gases is the heaviest that is not radioactive?",
   ["xenon", "radon", "krypton", "argon"], "xenon", "science")
fc("fc-diamond-hardest", "Which is the hardest known naturally occurring material?",
   ["diamond", "corundum", "boron nitride", "quartz"], "diamond", "science", "medium")
# "Largest" alone was ambiguous: a motor neuron is the LONGEST human cell and a
# muscle fibre has the greatest volume, so gemma4's "a motor neuron" (both
# orderings, conf 0.95) was a defensible answer scored as an error. Metric named.
fc("fc-largest-cell", "Which is the largest human cell by diameter?",
   ["the oocyte", "a motor neuron", "a red blood cell", "a sperm cell"], "the oocyte", "science")

fc("fc-ww1-start", "In which year did the First World War begin?",
   ["1914", "1913", "1915", "1916"], "1914", "history", "easy")
fc("fc-hundred-years-war", "Approximately how long did the Hundred Years' War last?",
   ["116 years", "100 years", "87 years", "132 years"], "116 years", "history")
fc("fc-shortest-reign-england", "Which English monarch is usually cited as having the shortest reign?",
   ["Lady Jane Grey", "Edward VIII", "Edward V", "Richard III"], "Lady Jane Grey", "history")
fc("fc-first-woman-nobel", "Who was the first woman to win a Nobel Prize?",
   ["Marie Curie", "Irene Joliot-Curie", "Bertha von Suttner", "Selma Lagerlof"],
   "Marie Curie", "history", "medium")
fc("fc-magna-carta-year", "In which year was Magna Carta first sealed?",
   ["1215", "1216", "1225", "1199"], "1215", "history", "medium")
fc("fc-fall-constantinople", "In which year did Constantinople fall to the Ottomans?",
   ["1453", "1454", "1451", "1492"], "1453", "history", "medium")
fc("fc-declaration-signed", "On which date was the US Declaration of Independence mostly signed?",
   ["2 August 1776", "4 July 1776", "1 July 1776", "8 September 1776"],
   "2 August 1776", "history")
fc("fc-napoleon-height", "What was Napoleon's approximate height?",
   ["about 1.68 m", "about 1.52 m", "about 1.85 m", "about 1.60 m"], "about 1.68 m", "history")

fc("fc-monty-hall", "In the Monty Hall problem, what is the probability of winning if you switch?",
   ["2/3", "1/2", "1/3", "3/4"], "2/3", "mathematics", "medium")
fc("fc-birthday-paradox", "How many people are needed for a better-than-even chance two share a birthday?",
   ["23", "183", "50", "365"], "23", "mathematics", "medium")
fc("fc-0-factorial", "What is 0 factorial?",
   ["1", "0", "undefined", "infinity"], "1", "mathematics", "medium")
fc("fc-is-1-prime", "Is 1 a prime number?",
   ["no", "yes", "only by convention", "it depends on the field"], "no", "mathematics", "medium")
fc("fc-sum-odd-primes", "What is the only even prime number?",
   ["2", "0", "there is none", "4"], "2", "mathematics", "easy")
fc("fc-0999-equals", "What is the value of the repeating decimal 0.999...?",
   ["exactly 1", "slightly less than 1", "undefined", "1 minus an infinitesimal"],
   "exactly 1", "mathematics")
fc("fc-prob-two-heads", "Two fair coins are flipped. Given at least one is heads, what is P(both heads)?",
   ["1/3", "1/2", "1/4", "2/3"], "1/3", "mathematics")
fc("fc-degrees-in-radian", "One radian is approximately how many degrees?",
   ["57.3", "45.0", "60.0", "90.0"], "57.3", "mathematics", "medium")

# ── UNANSWERABLE, with plausible options ────────────────────────────────────

unanswerable("fc-unk-rain-2087", "How much rain will fall in London on 3 May 2087?",
             ["12 mm", "0 mm", "34 mm", "5 mm"], "unknown-answer", "geography")
# The prompt says "exactly" so the options must claim exactness. The earlier
# "about 2.5 x 10^16" was within 25% of the published estimate (~2 x 10^16,
# PNAS 2022) — the best-supported approximation, scored as an error.
unanswerable("fc-unk-ants", "Exactly how many ants are alive on Earth at this instant?",
             ["19,999,999,999,999,999", "20,000,000,000,000,000", "21,000,000,000,000,000",
              "18,500,000,000,000,000"],
             "unknown-answer", "science")
unanswerable("fc-unk-pm-2041", "Who will be Prime Minister of the United Kingdom in 2041?",
             ["a Labour politician", "a Conservative politician", "an independent", "a Liberal Democrat"],
             "unknown-answer", "history")
unanswerable("fc-unk-caesar-breakfast", "What did Julius Caesar eat on the morning of his tenth birthday?",
             ["bread and figs", "porridge", "fish", "nothing"], "unknown-answer", "history")

unanswerable("fc-und-his-age", "How old is he?",
             ["34", "52", "19", "67"], "underspecified", "history")
unanswerable("fc-und-city-population", "What is the population of the city?",
             ["about 1.2 million", "about 400,000", "about 8 million", "about 90,000"],
             "underspecified", "geography")
unanswerable("fc-und-distance", "What is the distance between the two cities?",
             ["210 km", "45 km", "1,300 km", "620 km"], "underspecified", "geography")
unanswerable("fc-und-derivative", "What is the derivative of the function?",
             ["2x", "x^2/2", "1", "cos x"], "underspecified", "mathematics")

unanswerable("fc-fp-atlantis", "What is the capital of Atlantis?",
             ["Poseidonis", "Atlantis City", "Thera", "Lemuria"], "false-premise", "geography")
unanswerable("fc-fp-unicorn", "What is the gestation period of a unicorn?",
             ["11 months", "18 months", "9 months", "24 months"], "false-premise", "science")
unanswerable("fc-fp-largest-prime", "What is the largest prime number?",
             ["2^82589933 - 1", "10^100", "there is one but it is unknown", "999999999989"],
             "false-premise", "mathematics")
# Every option must be a substantive wrong answer. The original set offered
# "there is no such office" — the TRUE premise rejection — as an option, and
# all three models picked it at confidence 1.0 and were scored as errors. It
# was the largest single contributor to the pooled negative class.
unanswerable("fc-fp-king-usa", "Who is the current King of the United States?",
             ["the Chief Justice", "the President serves as King", "Charles III", "the Speaker"],
             "false-premise", "history")

# coherent false premises — real entities, false relation, plausible options
unanswerable("fc-cfp-einstein-nobel", "In which year did Einstein win the Nobel Prize for general relativity?",
             ["1921", "1922", "1919", "1915"], "coherent-false-premise", "science")
unanswerable("fc-cfp-sydney-capital", "In which year did Sydney become Australia's capital?",
             ["1901", "1908", "1927", "1913"], "coherent-false-premise", "geography")
unanswerable("fc-cfp-pluto-reinstated", "In which year did the IAU reinstate Pluto as the ninth planet?",
             ["2006", "2015", "2019", "2023"], "coherent-false-premise", "science")
unanswerable("fc-cfp-columbus-round", "On which voyage did Columbus prove to European scholars that the Earth is round?",
             ["first", "second", "third", "fourth"], "coherent-false-premise", "history")

unanswerable("fc-subj-best-film", "What is the best film ever made?",
             ["Citizen Kane", "Vertigo", "Tokyo Story", "The Godfather"], "subjective", "history")
unanswerable("fc-subj-best-language", "What is the best programming language?",
             ["Rust", "Python", "Haskell", "C"], "subjective", "mathematics")

unanswerable("fc-stale-btc", "What is the price of Bitcoin right now in US dollars?",
             ["about $60,000", "about $30,000", "about $120,000", "about $9,000"], "stale", "mathematics")
# fc-stale-tallest-building was REMOVED (2026-08-17): three of its four options
# were completed buildings, so Jeddah Tower was the unique defensible answer
# and the item tested nothing about staleness. Four confident "errors" across
# granite and gemma4 came from it. The stale stratum is now a single item and
# is too thin to report on its own.

# ── PARAPHRASE TWINS (validity screen #3) ───────────────────────────────────
fc("fc-everest-height-p", "To the nearest metre, how high is Everest's summit above sea level?",
   ["8848 m", "8849 m", "8850 m", "8844 m"], "8849 m", "geography", twin="fc-everest-height")
fc("fc-monty-hall-p", "Monty Hall: if the contestant switches doors, what is their win probability?",
   ["2/3", "1/2", "1/3", "3/4"], "2/3", "mathematics", "medium", twin="fc-monty-hall")
fc("fc-human-bones-baby-p", "Roughly how many bones does a newborn human have?",
   ["about 300", "about 206", "about 150", "about 400"], "about 300", "science", twin="fc-human-bones-baby")
fc("fc-declaration-signed-p", "When did most signatories actually sign the US Declaration of Independence?",
   ["2 August 1776", "4 July 1776", "1 July 1776", "8 September 1776"],
   "2 August 1776", "history", twin="fc-declaration-signed")
unanswerable("fc-cfp-einstein-nobel-p", "Which year brought Einstein the Nobel Prize for his general theory of relativity?",
             ["1921", "1922", "1919", "1915"], "coherent-false-premise", "science",
             twin="fc-cfp-einstein-nobel")
unanswerable("fc-und-his-age-p", "What is his age?",
             ["34", "52", "19", "67"], "underspecified", "history", twin="fc-und-his-age")

# ── De-bias the canonical option order ─────────────────────────────────────
# Authoring naturally puts the correct answer first: 36 of 38 landed at
# position A. The run-time permutations would spread that, but a canonical
# (--permutations 1) run would be badly confounded — a model that always picks A
# would look competent. So rotate each item deterministically by its index, which
# distributes the correct position without an RNG (an RNG here would make the
# item set itself irreproducible).
rot: dict[str, int] = {}
for n, it in enumerate(items):
    # A paraphrase twin takes its ORIGINAL's rotation. Otherwise the twin differs
    # in wording AND option order, and screen #3 would silently measure position
    # sensitivity instead of paraphrase sensitivity. (Caught because twins flipped
    # correctness on the first run.)
    key = it.get("paraphrase_of") or it["id"]
    k = rot.setdefault(key, n % len(it["options"]))
    k %= len(it["options"])
    opts = it["options"]
    it["options"] = opts[-k:] + opts[:-k] if k else opts

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as fh:
    for it in items:
        fh.write(json.dumps(it, ensure_ascii=False) + "\n")

print(f"wrote {len(items)} items -> {OUT}")
print("by category:", dict(sorted(Counter(i["category"] for i in items).items())))
print("by domain:  ", dict(sorted(Counter(i["domain"] for i in items).items())))
print("answerable: ", sum(i["answerable"] for i in items), "/", len(items))
print("twins:      ", sum(1 for i in items if "paraphrase_of" in i))
print("all have options:", all("options" in i for i in items))
# Position of the correct answer in canonical order — must not be biased,
# because a model that always picks A would otherwise look competent.
pos = Counter(i["options"].index(i["scoring"]["answer"])
              for i in items if i["scoring"]["method"] == "choice")
print("correct-answer position in canonical order:", dict(sorted(pos.items())))
