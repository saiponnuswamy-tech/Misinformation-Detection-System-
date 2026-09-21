"""
detector.py — core analysis engine for the misinformation detection system.

Pure standard library, so it runs with no downloads and no GPU. Every signal is
explicit and inspectable, which is what makes the output explainable.

Pipeline
    1. clean()            normalise the raw text
    2. split_sentences()  sentence segmentation
    3. extract_claims()   pick out check-worthy factual statements
    4. run_signals()      fire the linguistic / source / context detectors
    5. score()            fuse signal weights into a 0-100 risk score
    6. analyse()          wrap it all into one explainable report

Swapping in a transformer (BERT / RoBERTa) later means replacing the body of
extract_claims() and adding a model score into score() as one more weighted
signal — the report format does not change.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

# --------------------------------------------------------------------------
# Lexicons. Kept as data, not code, so they can be extended per language or
# platform without touching the scoring logic.
# --------------------------------------------------------------------------

SENSATIONAL = [
    "shocking", "shocked", "bombshell", "explosive", "unbelievable", "insane",
    "mind-blowing", "miracle", "horrifying", "terrifying", "outrageous",
    "you won't believe", "jaw-dropping", "stunning revelation", "gone viral",
]

ABSOLUTE = [
    "always", "never", "everyone knows", "nobody", "all doctors", "100%",
    "completely safe", "totally cured", "guaranteed", "proven fact",
    "without a doubt", "undeniable", "the only",
]

URGENCY = [
    "forward this", "share before", "share this with everyone", "act now",
    "delete", "before it's removed", "before they take it down", "urgent",
    "immediately", "spread the word", "send to all your", "last chance",
    "breaking", "right now",
]

CONSPIRACY = [
    "they don't want you to know", "they dont want you to know", "wake up",
    "cover up", "coverup", "mainstream media", "the truth they hide",
    "big pharma", "deep state", "hidden agenda", "censored", "suppressed",
    "do your own research", "what they aren't telling you",
]

VAGUE_ATTRIBUTION = [
    "scientists say", "experts say", "doctors say", "studies show",
    "research shows", "a study found", "sources say", "it is said",
    "people are saying", "reports suggest", "a doctor from", "insiders claim",
]

HEALTH_CURE = [
    "cure", "cures", "cured", "heals", "detox", "miracle remedy",
    "kills the virus", "prevents cancer", "natural remedy beats",
    "boost immunity instantly", "no side effects",
]

FINANCIAL_BAIT = [
    "double your money", "guaranteed returns", "risk free", "risk-free",
    "get rich", "limited slots", "investment opportunity of a lifetime",
    "crypto giveaway", "claim your reward",
]

# Signals that push the score *down* — markers of careful reporting.
CREDIBLE_MARKERS = [
    "according to", "published in", "peer-reviewed", "the study of",
    "data from", "on record", "spokesperson", "press release",
    "official statement", "as reported by", "court filing", "quoted as saying",
]

HEDGES = ["may", "might", "could", "suggests", "appears to", "reportedly", "alleged"]

TRUSTED_DOMAINS = {
    "who.int", "nature.com", "science.org", "nih.gov", "cdc.gov", "gov.in",
    "pib.gov.in", "reuters.com", "apnews.com", "bbc.co.uk", "bbc.com",
    "thehindu.com", "indianexpress.com", "nytimes.com", "pubmed.ncbi.nlm.nih.gov",
}

SUSPICIOUS_TLDS = (".xyz", ".top", ".click", ".info", ".buzz", ".tk", ".ml")

URL_RE = re.compile(r"https?://([\w.-]+)[^\s]*", re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:%|percent|crore|lakh|million|billion|kg|km|times)?\b", re.I)
OPINION_RE = re.compile(r"\b(i think|i feel|in my opinion|imo|i believe|personally)\b", re.I)


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class Signal:
    """One fired detector: what it is, how much it weighs, and the evidence."""
    code: str
    category: str           # language | source | context
    label: str
    weight: float           # positive = raises risk, negative = lowers it
    evidence: List[str] = field(default_factory=list)

    def explain(self) -> str:
        if not self.evidence:
            return self.label
        shown = ", ".join(f'"{e}"' for e in self.evidence[:3])
        return f"{self.label} ({shown})"


@dataclass
class Claim:
    text: str
    checkworthiness: float  # 0..1
    reasons: List[str]


# --------------------------------------------------------------------------
# 1-2. Preprocessing
# --------------------------------------------------------------------------

def clean(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    rough = SENT_SPLIT_RE.split(text)
    out: List[str] = []
    for part in rough:
        for line in part.split("\n"):
            line = line.strip()
            if len(line) > 1:
                out.append(line)
    return out


def _find(text_low: str, phrases: List[str]) -> List[str]:
    return [p for p in phrases if p in text_low]


# --------------------------------------------------------------------------
# 3. Claim extraction
# --------------------------------------------------------------------------

def extract_claims(sentences: List[str], top_k: int = 5) -> List[Claim]:
    """Rank sentences by how check-worthy they are.

    A check-worthy sentence asserts something about the world that could in
    principle be verified: it carries a number, a named entity, or a factual
    verb, and it is not flagged as personal opinion or a question.
    """
    claims: List[Claim] = []
    for s in sentences:
        low = s.lower()
        score, reasons = 0.0, []

        if NUMBER_RE.search(s):
            score += 0.35
            reasons.append("contains a figure or quantity")

        # Crude named-entity proxy: capitalised word that is not sentence-initial.
        caps = re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b", s)
        if caps:
            score += 0.25
            reasons.append("names a person, place or organisation")

        if re.search(r"\b(is|are|was|were|has|have|will|causes|caused|kills|cures|banned|approved|announced|found)\b", low):
            score += 0.25
            reasons.append("states a fact rather than an opinion")

        if _find(low, VAGUE_ATTRIBUTION):
            score += 0.15
            reasons.append("attributes the claim to an unnamed authority")

        if OPINION_RE.search(s):
            score -= 0.45
            reasons.append("framed as personal opinion")

        if s.rstrip().endswith("?"):
            score -= 0.30
            reasons.append("phrased as a question")

        if len(s.split()) < 4:
            score -= 0.25

        score = max(0.0, min(1.0, score))
        if score >= 0.4:
            claims.append(Claim(text=s, checkworthiness=round(score, 2), reasons=reasons))

    claims.sort(key=lambda c: c.checkworthiness, reverse=True)
    return claims[:top_k]


# --------------------------------------------------------------------------
# 4. Signal detectors
# --------------------------------------------------------------------------

def run_signals(text: str) -> List[Signal]:
    low = text.lower()
    words = text.split()
    signals: List[Signal] = []

    def add(code, cat, label, weight, evidence=None):
        signals.append(Signal(code, cat, label, weight, evidence or []))

    hits = _find(low, SENSATIONAL)
    if hits:
        add("SENSATIONAL", "language",
            "Sensational or emotionally loaded wording", 12 + 3 * (len(hits) - 1), hits)

    hits = _find(low, ABSOLUTE)
    if hits:
        add("ABSOLUTE", "language",
            "Absolute claims stated without qualification", 10 + 3 * (len(hits) - 1), hits)

    hits = _find(low, URGENCY)
    if hits:
        add("URGENCY", "language",
            "Pressure to share or act immediately", 14 + 4 * (len(hits) - 1), hits)

    hits = _find(low, CONSPIRACY)
    if hits:
        add("CONSPIRACY", "context",
            "Conspiratorial framing about hidden or suppressed truth", 16, hits)

    hits = _find(low, VAGUE_ATTRIBUTION)
    if hits:
        add("VAGUE_SOURCE", "source",
            "Claim attributed to an authority that is never named", 13, hits)

    hits = _find(low, HEALTH_CURE)
    if hits:
        add("HEALTH_CLAIM", "context",
            "Unverified medical or cure claim", 15, hits)

    hits = _find(low, FINANCIAL_BAIT)
    if hits:
        add("FINANCIAL_BAIT", "context",
            "Financial promise typical of scam messages", 18, hits)

    # Punctuation intensity
    bangs = text.count("!")
    if bangs >= 3:
        add("PUNCTUATION", "language",
            f"Excessive exclamation marks ({bangs})", min(4 + 2 * bangs, 12))

    # Shouting
    shouty = [w for w in words if len(w) >= 4 and w.isupper()]
    if len(shouty) >= 3:
        add("ALLCAPS", "language",
            f"Shouted words in capitals ({len(shouty)})", min(4 + 2 * len(shouty), 12),
            shouty[:4])

    # Numbers with no source
    nums = NUMBER_RE.findall(text)
    if nums and not _find(low, CREDIBLE_MARKERS) and not URL_RE.search(text):
        add("UNSOURCED_STAT", "source",
            "Figures given with no source, date or reference", 11, [n.strip() for n in nums[:3]])

    # Links
    domains = [d.lower().lstrip("www.") for d in URL_RE.findall(text)]
    if domains:
        trusted = [d for d in domains if any(d.endswith(t) for t in TRUSTED_DOMAINS)]
        shady = [d for d in domains if d.endswith(SUSPICIOUS_TLDS)]
        if trusted:
            add("TRUSTED_LINK", "source",
                "Links to a recognised, high-reputation source", -14, trusted)
        if shady:
            add("SHADY_DOMAIN", "source",
                "Links to a domain type common in spam campaigns", 14, shady)
        if not trusted and not shady:
            add("UNKNOWN_DOMAIN", "source",
                "Links to a source with no established reputation", 6, domains[:3])
    else:
        add("NO_SOURCE", "source",
            "No link, citation or source of any kind is given", 9)

    # Credibility credits
    hits = _find(low, CREDIBLE_MARKERS)
    if hits:
        add("ATTRIBUTED", "source",
            "Statements are attributed to a named, checkable source", -12, hits)

    hits = _find(low, HEDGES)
    if len(hits) >= 2:
        add("HEDGED", "language",
            "Careful, hedged language rather than certainty", -7, hits)

    if re.search(r"\b(19|20)\d{2}\b", text) or re.search(r"\b\d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", low):
        add("DATED", "context", "Includes a specific date or year", -5)

    return signals


# --------------------------------------------------------------------------
# 5. Scoring
# --------------------------------------------------------------------------

def score(signals: List[Signal], text: str) -> int:
    """Fuse weights into 0-100 with a saturating curve.

    A logistic squash stops a long message from piling up an automatic 100
    just because it is long, and keeps a short scam text from scoring low.
    """
    raw = sum(s.weight for s in signals)

    # Short inputs carry less evidence, so pull them toward the middle.
    n_words = max(len(text.split()), 1)
    confidence = min(1.0, math.log(n_words + 1) / math.log(60))

    squashed = 100 / (1 + math.exp(-(raw - 25) / 18))
    adjusted = 50 + (squashed - 50) * (0.55 + 0.45 * confidence)
    return int(round(max(0, min(100, adjusted))))


def band(value: int) -> Dict[str, str]:
    if value <= 30:
        return {"name": "Likely reliable", "key": "low",
                "advice": "Nothing suspicious stood out. Still worth a quick source check before sharing."}
    if value <= 65:
        return {"name": "Needs caution", "key": "medium",
                "advice": "Parts of this are unsupported or missing context. Verify before you forward it."}
    return {"name": "High risk", "key": "high",
            "advice": "Strong markers of false or manipulative content. Do not forward this."}


# --------------------------------------------------------------------------
# 6. Public API
# --------------------------------------------------------------------------

def analyse(raw_text: str) -> Dict[str, Any]:
    text = clean(raw_text)
    if len(text.split()) < 3:
        return {"error": "Enter at least a full sentence to analyse."}

    sentences = split_sentences(text)
    claims = extract_claims(sentences)
    signals = run_signals(text)
    value = score(signals, text)
    b = band(value)

    risky = sorted([s for s in signals if s.weight > 0], key=lambda s: -s.weight)
    credits = sorted([s for s in signals if s.weight < 0], key=lambda s: s.weight)

    return {
        "risk_score": value,
        "band": b["name"],
        "band_key": b["key"],
        "advice": b["advice"],
        "word_count": len(text.split()),
        "sentence_count": len(sentences),
        "claims": [asdict(c) for c in claims],
        "warnings": [{"code": s.code, "category": s.category, "text": s.explain(),
                      "weight": round(s.weight, 1)} for s in risky],
        "positives": [{"code": s.code, "category": s.category, "text": s.explain(),
                       "weight": round(s.weight, 1)} for s in credits],
        "summary": _summarise(value, b, risky, claims),
    }


def _summarise(value, b, risky, claims) -> str:
    if not risky:
        return (f"Risk {value}/100 — {b['name'].lower()}. No suspicious language or "
                f"sourcing patterns were detected in this text.")
    top = "; ".join(w.label.lower() for w in risky[:3])
    claim_bit = f' The main checkable claim is: "{claims[0].text}"' if claims else ""
    return f"Risk {value}/100 — {b['name'].lower()}. Main reasons: {top}.{claim_bit}"


if __name__ == "__main__":
    import json
    import sys

    sample = " ".join(sys.argv[1:]) or (
        "SHOCKING!! Doctors say this miracle drink CURES diabetes in 3 days. "
        "Big pharma doesn't want you to know. 100% guaranteed, no side effects. "
        "Forward this to everyone before it is deleted!!!"
    )
    print(json.dumps(analyse(sample), indent=2))
