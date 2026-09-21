# Misinformation-Detection-System-
To develop an AI-powered system that detects and analyzes potentially misleading information shared through social media, messaging platforms, and online sources by evaluating claims, identifying misinformation patterns, and providing users with clear, explainable verification insights before they share the content.
# Claim Check — AI-Powered Misinformation Detection System

Paste a forwarded message, social post or headline. The system extracts the
checkable claims, tests the language and sourcing for known manipulation
patterns, and returns a **0–100 risk score with the reasons behind it**.

It is decision support, not censorship: nothing is blocked or deleted, and every
score comes with the evidence that produced it.

---

## Quick start

```bash
cd misinformation-detector
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>.

Run the engine straight from the command line, with no server:

```bash
python detector.py "SHOCKING!! Doctors say this drink CURES diabetes. Forward to everyone!!"
```

Run the tests:

```bash
python -m pytest -q          # or: python tests/test_detector.py
```

---

## Files

| File | What it holds |
|---|---|
| `detector.py` | The whole analysis engine — preprocessing, claim extraction, signal detectors, scoring. Pure standard library. |
| `app.py` | Flask server: web interface, JSON API, SQLite feedback store. |
| `templates/index.html` | The interface — input panel, risk dial, claim list, reason list. |
| `tests/test_detector.py` | Eight tests covering scoring, claim extraction and bounds. |
| `requirements.txt` | Dependencies, with the transformer upgrade path commented out. |

---

## How the pipeline works

```
raw text
   │
   ├─ clean()            normalise whitespace and unicode
   ├─ split_sentences()  sentence segmentation
   ├─ extract_claims()   rank sentences by check-worthiness
   ├─ run_signals()      fire ~14 detectors across three categories
   ├─ score()            weighted sum → logistic squash → 0-100
   └─ analyse()          assemble the explainable report
```

**Claim extraction.** A sentence is check-worthy when it asserts something about
the world: it carries a figure, a named entity, or a factual verb. Personal
opinion (`I think…`), questions, and very short fragments are pushed down.
Each claim comes back with a score from 0 to 1 and the reasons it was picked.

**Signals.** Each detector returns a weight. Positive weights raise risk,
negative weights lower it — so careful reporting is actively rewarded rather
than merely not penalised.

| Category | Examples | Weight |
|---|---|---|
| Language | sensational wording, absolute claims, urgency to forward, shouting, exclamation pile-up | +10 to +18 |
| Source | authority never named, figures with no reference, spam-typical domain, no citation at all | +6 to +14 |
| Context | conspiratorial framing, unverified cure claim, scam-style financial promise | +15 to +18 |
| Credits | links to a high-reputation domain, named attribution, hedged language, a specific date | −5 to −14 |

**Scoring.** The weights are summed, then pushed through a logistic curve
centred at 25 so a long message cannot pile up an automatic 100 and a short scam
text cannot slip through low. A confidence factor derived from word count pulls
very short inputs toward the middle, since three words carry little evidence.

| Score | Band | What the user is told |
|---|---|---|
| 0–30 | Likely reliable | Nothing suspicious stood out; still worth a source check |
| 31–65 | Needs caution | Unsupported or missing context; verify before forwarding |
| 66–100 | High risk | Strong markers of false or manipulative content; don't forward |

---

## API

**POST `/api/analyse`**

```bash
curl -X POST http://127.0.0.1:5000/api/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "SHOCKING!! Doctors say this drink CURES diabetes. Forward to everyone!!"}'
```

```json
{
  "risk_score": 87,
  "band": "High risk",
  "band_key": "high",
  "advice": "Strong markers of false or manipulative content. Do not forward this.",
  "claims": [
    { "text": "Doctors say this drink CURES diabetes.",
      "checkworthiness": 0.65,
      "reasons": ["states a fact rather than an opinion",
                  "attributes the claim to an unnamed authority"] }
  ],
  "warnings": [
    { "code": "URGENCY", "category": "language",
      "text": "Pressure to share or act immediately (\"forward\")", "weight": 14 }
  ],
  "positives": [],
  "elapsed_ms": 0.6
}
```

Pass `{"url": "https://…"}` instead of `text` to fetch and analyse a web page
(needs `requests` and `beautifulsoup4`).

**Other endpoints**

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/feedback` | Record `agree` / `disagree` on a verdict |
| GET | `/api/feedback/export` | Dump collected labels as JSON — the next training set |
| GET | `/api/health` | Liveness probe |

---

## Going further

The rule layer is deliberately explicit so every score can be explained. To add
a learned layer without changing the report format:

1. **Claim detection** — replace the body of `extract_claims()` with a
   fine-tuned BERT classifier over the ClaimBuster dataset. Keep returning the
   same `Claim` objects.
2. **Stance and veracity** — train a RoBERTa classifier on LIAR or FakeNewsNet,
   then append its output to `run_signals()` as one more weighted `Signal`
   (`code="MODEL"`, weight = `confidence × 25`). Scoring code needs no edit.
3. **Evidence retrieval** — embed claims with Sentence-BERT, search a fact-check
   corpus, and add a `CONTRADICTED` or `SUPPORTED` signal from the nearest match.
4. **Attribution** — run SHAP or LIME over the learned model and merge the top
   features into the existing reason list.
5. **Regional languages** — the lexicons in `detector.py` are plain lists.
   Add Hindi, Telugu or Tamil phrase lists and detect the language in `clean()`.

## Limitations, stated plainly

- Lexicons catch known phrasings; a carefully worded falsehood can score low.
- Satire and strong opinion writing can trip the language detectors.
- Source reputation uses a static domain list, not a live reputation service.
- A high score means *patterns common in false content*, never proof of falsity.
