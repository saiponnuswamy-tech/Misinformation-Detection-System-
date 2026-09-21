"""
app.py — Flask server for the misinformation detection system.

Run:
    pip install -r requirements.txt
    python app.py
    open http://127.0.0.1:5000

Endpoints
    GET  /                 the web interface
    POST /api/analyse      {"text": "..."} or {"url": "..."} -> full report
    POST /api/feedback     {"text": ..., "verdict": "agree"|"disagree"} -> stored
    GET  /api/health       liveness probe
"""

import json
import os
import sqlite3
import time
from urllib.parse import urlparse

from flask import Flask, jsonify, render_template, request

import detector

app = Flask(__name__)
DB_PATH = os.environ.get("DETECTOR_DB", "feedback.db")
MAX_CHARS = 20000


# --------------------------------------------------------------------------
# Feedback store — the beginning of the continuous-learning loop
# --------------------------------------------------------------------------

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                   id        INTEGER PRIMARY KEY AUTOINCREMENT,
                   created   REAL,
                   text      TEXT,
                   score     INTEGER,
                   verdict   TEXT,
                   comment   TEXT
               )"""
        )


def fetch_article(url: str) -> str:
    """Pull the visible text of a page. Optional — needs requests + bs4."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("URL fetching needs: pip install requests beautifulsoup4")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https URLs are supported.")

    html = requests.get(url, timeout=10, headers={"User-Agent": "MisinfoDetector/1.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    body = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    return f"{title}. {body}".strip()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", version="1.0")


@app.post("/api/analyse")
def analyse():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    url = (payload.get("url") or "").strip()

    if url and not text:
        try:
            text = fetch_article(url)
        except Exception as exc:
            return jsonify(error=str(exc)), 400

    if not text:
        return jsonify(error="Send either 'text' or 'url'."), 400
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    started = time.perf_counter()
    report = detector.analyse(text)
    if "error" in report:
        return jsonify(report), 400

    report["source_url"] = url or None
    report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return jsonify(report)


@app.post("/api/feedback")
def feedback():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "")[:MAX_CHARS]
    verdict = payload.get("verdict")
    if verdict not in ("agree", "disagree"):
        return jsonify(error="verdict must be 'agree' or 'disagree'."), 400

    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO feedback (created, text, score, verdict, comment) VALUES (?,?,?,?,?)",
            (time.time(), text, payload.get("score"), verdict, payload.get("comment", "")),
        )
    return jsonify(status="recorded")


@app.get("/api/feedback/export")
def export_feedback():
    """Dump collected labels as JSON — the training set for the next model."""
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT created, text, score, verdict, comment FROM feedback ORDER BY id DESC"
        ).fetchall()
    keys = ("created", "text", "score", "verdict", "comment")
    return app.response_class(
        json.dumps([dict(zip(keys, r)) for r in rows], indent=2),
        mimetype="application/json",
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
