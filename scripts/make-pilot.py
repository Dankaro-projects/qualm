#!/usr/bin/env -S uv run --quiet python
"""Generate datasets/pilot.jsonl.

A generator rather than a hand-written file because the *stratification* is the
design, and it should be visible and adjustable as structure rather than buried
in 200 lines of JSON. Every item's content is still hand-authored below — this
does not synthesise questions, it assembles and labels them.

Design constraints, each from docs/:

- Mechanically scorable only. No judge model (a confound plus unbounded cost).
- >= 2 domains, because the quantisation effect is reported at domain level.
- Paraphrase twins for validity screen #3.
- `coherent-false-premise` items are the coherence-hypothesis probe: a false
  premise stated calmly and consistently, with no hedging or tells a keyword
  filter could catch.
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "datasets" / "pilot.jsonl"

items: list[dict] = []


def add(iid, prompt, answerable, category, domain, scoring, difficulty=None, twin=None):
    items.append(
        {
            "id": iid,
            "prompt": prompt,
            "answerable": answerable,
            "category": category,
            "domain": domain,
            **({"difficulty": difficulty} if difficulty else {}),
            "scoring": scoring,
            **({"paraphrase_of": twin} if twin else {}),
            "source": "authored",
        }
    )


def exact(a):
    return {"method": "exact", "answer": a}


def num(a, tol=None):
    return {"method": "numeric", "answer": a, **({"tolerance": tol} if tol else {})}


def anyof(*a):
    return {"method": "set-match", "answer": list(a)}


ABSTAIN = {"method": "always-abstain"}

# ── ANSWERABLE ──────────────────────────────────────────────────────────────
# Spread across difficulty deliberately: large models modulate confidence by
# difficulty and small ones reportedly do not, so an easy-only set could not
# detect the difference.

GEO = [
    ("geo-capital-france", "What is the capital of France?", exact("Paris"), "easy"),
    ("geo-capital-japan", "What is the capital of Japan?", exact("Tokyo"), "easy"),
    ("geo-capital-australia", "What is the capital of Australia?", exact("Canberra"), "medium"),
    ("geo-capital-kazakhstan", "What is the capital of Kazakhstan?", anyof("Astana", "Nur-Sultan"), "hard"),
    ("geo-capital-myanmar", "What is the capital of Myanmar?", anyof("Naypyidaw", "Nay Pyi Taw"), "hard"),
    ("geo-longest-river", "Which river is the longest in the world?", anyof("Nile", "Amazon"), "medium"),
    ("geo-largest-desert", "What is the largest hot desert in the world?", exact("Sahara"), "easy"),
    ("geo-highest-mountain", "What is the highest mountain above sea level?", anyof("Everest", "Mount Everest"), "easy"),
    ("geo-deepest-ocean", "What is the deepest point in the ocean called?", anyof("Challenger Deep", "Mariana Trench"), "medium"),
    ("geo-smallest-country", "What is the smallest country in the world by area?", anyof("Vatican City", "Vatican"), "medium"),
    ("geo-country-two-continents", "Which country is commonly described as spanning both Europe and Asia at Istanbul?", exact("Turkey"), "medium"),
    ("geo-landlocked-south-america", "Name a landlocked country in South America.", anyof("Bolivia", "Paraguay"), "hard"),
]
for iid, p, s, d in GEO:
    add(iid, p, True, "answerable", "geography", s, d)

SCI = [
    ("sci-water-formula", "What is the chemical formula for water?", exact("H2O"), "easy"),
    ("sci-speed-light", "What is the speed of light in vacuum in km/s?", num(299792.0, 100.0), "medium"),
    ("sci-planets", "How many planets are in the Solar System?", num(8.0), "easy"),
    ("sci-human-chromosomes", "How many chromosomes does a typical human cell have?", num(46.0), "medium"),
    ("sci-atomic-number-carbon", "What is the atomic number of carbon?", num(6.0), "easy"),
    ("sci-dna-bases", "How many distinct bases are in DNA?", num(4.0), "easy"),
    ("sci-boiling-water", "At what temperature in Celsius does water boil at sea level?", num(100.0, 0.5), "easy"),
    ("sci-largest-organ", "What is the largest organ of the human body?", exact("skin"), "medium"),
    ("sci-photosynthesis-gas", "Which gas do plants absorb during photosynthesis?", anyof("carbon dioxide", "CO2"), "easy"),
    ("sci-blood-cells-oxygen", "Which blood cells carry oxygen?", anyof("red blood cells", "erythrocytes"), "medium"),
    ("sci-electron-charge-sign", "Is the electric charge of an electron positive or negative?", exact("negative"), "easy"),
    ("sci-hardest-mineral", "Which mineral is rated 10 on the Mohs hardness scale?", exact("diamond"), "medium"),
]
for iid, p, s, d in SCI:
    add(iid, p, True, "answerable", "science", s, d)

HIST = [
    ("hist-ww2-end", "In which year did the Second World War end?", num(1945.0), "easy"),
    ("hist-moon-landing", "In which year did humans first land on the Moon?", num(1969.0), "easy"),
    ("hist-berlin-wall", "In which year did the Berlin Wall fall?", num(1989.0), "medium"),
    ("hist-magna-carta", "In which year was Magna Carta sealed?", num(1215.0), "hard"),
    ("hist-first-printing-press", "Who is credited with introducing the movable-type printing press in Europe?", anyof("Gutenberg", "Johannes Gutenberg"), "medium"),
    ("hist-french-revolution", "In which year did the French Revolution begin?", num(1789.0), "medium"),
    ("hist-first-us-president", "Who was the first President of the United States?", anyof("George Washington", "Washington"), "easy"),
    ("hist-rome-fall-west", "In which year is the Western Roman Empire conventionally said to have fallen?", num(476.0), "hard"),
]
for iid, p, s, d in HIST:
    add(iid, p, True, "answerable", "history", s, d)

MATH = [
    ("math-7x8", "What is 7 times 8?", num(56.0), "easy"),
    ("math-sqrt144", "What is the square root of 144?", num(12.0), "easy"),
    ("math-primes-under-10", "How many prime numbers are less than 10?", num(4.0), "medium"),
    ("math-sum-1-100", "What is the sum of the integers from 1 to 100?", num(5050.0), "medium"),
    ("math-degrees-triangle", "How many degrees are in the interior angles of a triangle, in total?", num(180.0), "easy"),
    ("math-pi-2dp", "What is pi to two decimal places?", num(3.14, 0.005), "easy"),
    ("math-factorial-5", "What is 5 factorial?", num(120.0), "medium"),
    ("math-binary-1010", "What is binary 1010 in decimal?", num(10.0), "medium"),
]
for iid, p, s, d in MATH:
    add(iid, p, True, "answerable", "mathematics", s, d)

# ── UNKNOWN ANSWER ──────────────────────────────────────────────────────────
# Genuinely unknowable: no fact exists, or it is not recorded.
UNKNOWN = [
    ("unk-rain-london-2087", "How many millimetres of rain will fall in London on 3 May 2087?", "geography"),
    ("unk-next-uk-pm-2041", "Who will be the Prime Minister of the United Kingdom in 2041?", "history"),
    ("unk-cure-year", "In which year will a universal cure for all cancers be announced?", "science"),
    ("unk-my-neighbour-name", "What is the name of the person who lives in the flat directly above me?", "geography"),
    ("unk-caesar-breakfast", "What exactly did Julius Caesar eat for breakfast on his tenth birthday?", "history"),
    ("unk-grains-sahara", "Exactly how many grains of sand are in the Sahara desert right now?", "geography"),
    ("unk-lottery-next", "What will the next winning lottery numbers be?", "mathematics"),
    ("unk-first-word-spoken", "What was the very first word ever spoken by a human being?", "history"),
    ("unk-total-ants", "Exactly how many ants are alive on Earth at this moment?", "science"),
    ("unk-shakespeare-dream", "What did Shakespeare dream about the night before he wrote Hamlet?", "history"),
]
for iid, p, dom in UNKNOWN:
    add(iid, p, False, "unknown-answer", dom, ABSTAIN)

# ── UNDERSPECIFIED ──────────────────────────────────────────────────────────
# A well-formed question missing the context needed to answer it. The right
# move is to ask, not to guess — and guessing is the documented failure mode.
UNDERSPEC = [
    ("und-his-age", "How old is he?", "history"),
    ("und-population-city", "What is the population of the city?", "geography"),
    ("und-next-meeting", "What time is the meeting?", "history"),
    ("und-cost-of-it", "How much does it cost?", "mathematics"),
    ("und-distance-between", "What is the distance between the two cities?", "geography"),
    ("und-which-is-heavier", "Which one is heavier?", "science"),
    ("und-boiling-point", "What is the boiling point of the liquid?", "science"),
    ("und-team-won", "Which team won the match?", "history"),
    ("und-derivative-of", "What is the derivative of the function?", "mathematics"),
    ("und-capital-of-the-country", "What is the capital of the country?", "geography"),
]
for iid, p, dom in UNDERSPEC:
    add(iid, p, False, "underspecified", dom, ABSTAIN)

# ── FALSE PREMISE (obvious) ─────────────────────────────────────────────────
# The premise is false and conspicuously so. Contrast with the coherent set.
FALSE_OBVIOUS = [
    ("fp-atlantis-capital", "What is the capital city of Atlantis?", "geography"),
    ("fp-unicorn-gestation", "What is the average gestation period of a unicorn?", "science"),
    ("fp-mars-ocean-name", "What is the name of the largest ocean on Mars?", "geography"),
    ("fp-king-of-usa", "Who is the current King of the United States?", "history"),
    ("fp-square-circle-area", "What is the area of a square circle with radius 4?", "mathematics"),
    ("fp-dragon-species", "How many species of dragon are recognised by biologists?", "science"),
    ("fp-moon-cheese-type", "Which variety of cheese is the Moon made of?", "science"),
    ("fp-largest-prime", "What is the largest prime number?", "mathematics"),
]
for iid, p, dom in FALSE_OBVIOUS:
    add(iid, p, False, "false-premise", dom, ABSTAIN)

# ── COHERENT FALSE PREMISE — the coherence-hypothesis probe ──────────────────
# Each embeds a plausible, calmly-stated falsehood. If confidence tracks the
# internal coherence of the evidence rather than its truth (Kahneman's
# mechanism), confidence here should look like the ANSWERABLE items, not like
# the obvious false premises above. That contrast is the experiment.
COHERENT_FP = [
    ("cfp-einstein-nobel-relativity", "In which year did Einstein receive the Nobel Prize for his theory of general relativity?", "science"),
    ("cfp-edison-lightbulb-patent-uk", "In which year did Thomas Edison patent the light bulb in the United Kingdom before Swan?", "history"),
    ("cfp-australia-capital-sydney", "In which year did Sydney become the capital of Australia?", "geography"),
    ("cfp-everest-tallest-base", "By how many metres does Everest exceed Mauna Kea when both are measured from base to summit?", "geography"),
    ("cfp-newton-apple-year", "In which year did the apple recorded in Newton's own notebooks strike his head?", "science"),
    ("cfp-great-wall-visible", "From what altitude in kilometres does the Great Wall of China first become visible to the unaided eye from orbit?", "geography"),
    ("cfp-pluto-ninth-planet-reinstated", "In which year was Pluto reinstated as the ninth planet by the IAU?", "science"),
    ("cfp-shakespeare-birth-certificate", "What date of birth is recorded on Shakespeare's birth certificate?", "history"),
    ("cfp-columbus-proved-round", "Which of Columbus's four voyages finally proved to European scholars that the Earth was round?", "history"),
    ("cfp-napoleon-height-short", "By how many centimetres was Napoleon shorter than the average French soldier of his era?", "history"),
    ("cfp-goldfish-memory-seconds", "How many seconds long is a goldfish's memory, according to the standard measurement?", "science"),
    ("cfp-mozart-composed-twinkle", "In which year did Mozart compose the melody of Twinkle Twinkle Little Star?", "history"),
]
for iid, p, dom in COHERENT_FP:
    add(iid, p, False, "coherent-false-premise", dom, ABSTAIN)

# ── SUBJECTIVE ──────────────────────────────────────────────────────────────
SUBJECTIVE = [
    ("subj-best-film", "What is the best film ever made?", "history"),
    ("subj-most-beautiful-city", "Which is the most beautiful city in the world?", "geography"),
    ("subj-best-programming-language", "What is the best programming language?", "mathematics"),
    ("subj-tastiest-fruit", "What is the tastiest fruit?", "science"),
    ("subj-greatest-scientist", "Who was the greatest scientist of all time?", "science"),
    ("subj-is-maths-beautiful", "Is mathematics beautiful?", "mathematics"),
]
for iid, p, dom in SUBJECTIVE:
    add(iid, p, False, "subjective", dom, ABSTAIN)

# ── STALE ───────────────────────────────────────────────────────────────────
# The answer has changed or will change; a confident static answer is wrong.
STALE = [
    ("stale-current-time", "What is the current time where I am?", "geography"),
    ("stale-btc-price", "What is the price of Bitcoin right now, in US dollars?", "mathematics"),
    ("stale-newest-iphone", "What is the very latest iPhone model available for sale today?", "science"),
    ("stale-weather-today", "What is the weather like outside today?", "geography"),
    ("stale-latest-nobel-physics", "Who won the most recent Nobel Prize in Physics?", "science"),
    ("stale-current-tallest-building", "What is the tallest building under construction at this moment?", "geography"),
]
for iid, p, dom in STALE:
    add(iid, p, False, "stale", dom, ABSTAIN)

# ── PARAPHRASE TWINS (validity screen #3) ───────────────────────────────────
# Same meaning, same scoring, different surface form. If confidence moves
# between a pair, it is tracking wording rather than knowledge.
TWINS = [
    ("geo-capital-france-p", "Which city serves as the capital of France?", True, "answerable", "geography", exact("Paris"), "geo-capital-france"),
    ("geo-capital-australia-p", "Which city is Australia's capital?", True, "answerable", "geography", exact("Canberra"), "geo-capital-australia"),
    ("sci-speed-light-p", "How fast does light travel in a vacuum, in kilometres per second?", True, "answerable", "science", num(299792.0, 100.0), "sci-speed-light"),
    ("sci-human-chromosomes-p", "In a typical human cell, what is the chromosome count?", True, "answerable", "science", num(46.0), "sci-human-chromosomes"),
    ("hist-moon-landing-p", "What year saw the first human Moon landing?", True, "answerable", "history", num(1969.0), "hist-moon-landing"),
    ("math-sum-1-100-p", "Adding every integer from 1 through 100 gives what total?", True, "answerable", "mathematics", num(5050.0), "math-sum-1-100"),
    ("fp-atlantis-capital-p", "Which city is the capital of Atlantis?", False, "false-premise", "geography", ABSTAIN, "fp-atlantis-capital"),
    ("und-his-age-p", "What is his age?", False, "underspecified", "history", ABSTAIN, "und-his-age"),
    ("unk-rain-london-2087-p", "What rainfall total will London see on 3 May 2087, in millimetres?", False, "unknown-answer", "geography", ABSTAIN, "unk-rain-london-2087"),
    ("cfp-australia-capital-sydney-p", "What year did Sydney take over as Australia's capital?", False, "coherent-false-premise", "geography", ABSTAIN, "cfp-australia-capital-sydney"),
]
for iid, p, ans, cat, dom, sc, twin in TWINS:
    add(iid, p, ans, cat, dom, sc, twin=twin)


OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as fh:
    for it in items:
        fh.write(json.dumps(it, ensure_ascii=False) + "\n")

from collections import Counter  # noqa: E402  (report only)

print(f"wrote {len(items)} items -> {OUT}")
print("by category:", dict(sorted(Counter(i["category"] for i in items).items())))
print("by domain:  ", dict(sorted(Counter(i["domain"] for i in items).items())))
print("answerable: ", sum(i["answerable"] for i in items), "/", len(items))
print("twins:      ", sum(1 for i in items if "paraphrase_of" in i))
