from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import feedparser
import requests

# Public energy / market RSS feeds (no auth required)
FEEDS = {
    "EnergyWatch": "https://energywatch.com/rss",
    "Euractiv Energy": "https://www.euractiv.com/sections/energy/feed/",
    "Reuters Energy (via Google News)":
        "https://news.google.com/rss/search?q=european+energy+gas+power+prices&hl=en-US&gl=US&ceid=US:en",
    "ENTSO-E News (via Google News)":
        "https://news.google.com/rss/search?q=ENTSO-E+OR+%22day-ahead%22+electricity+europe&hl=en-US&gl=US&ceid=US:en",
}

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "gemini-2.0-flash:generateContent")

PROMPT = """You are a junior market analyst at a European energy trading firm.
Below are recent news headlines. For each relevant one, extract a structured signal.
Ignore items unrelated to European gas, power, or carbon markets.

Direction conventions (direction refers to the PRICE of the tagged commodity):
- Supply reduction or demand increase = bullish. Supply increase or demand reduction = bearish.
- More renewable generation (wind/solar/hydro) = bearish for power prices.
- Higher gas prices = bullish for power prices (gas often sets the marginal price).
- Policies restricting fossil supply = bullish for that commodity; policies adding supply = bearish.
- The rationale must logically match the direction. If unsure, use "neutral".

Return ONLY a JSON array (no markdown fences, no commentary). Each element:
{
  "headline": "<original headline>",
  "source": "<source>",
  "commodity": "power" | "gas" | "carbon" | "other",
  "region": "<country or 'EU'>",
  "direction": "bullish" | "bearish" | "neutral",
  "impact": 1 | 2 | 3,            // 1=minor, 3=major price driver
  "rationale": "<one sentence: why this moves prices>"
}

Headlines:
{headlines}
"""


def fetch_headlines(max_per_feed: int = 8) -> list[dict]:
    """Pull recent items from all feeds. Returns [{title, source, link, published}]."""
    items = []
    for source, url in FEEDS.items():
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:max_per_feed]:
                items.append({
                    "title": re.sub(r"\s+", " ", entry.get("title", "")).strip(),
                    "source": source,
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception:
            continue  # a dead feed should never break the dashboard
    # de-duplicate near-identical titles
    seen, unique = set(), []
    for it in items:
        key = it["title"].lower()[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(it)
    return unique


def structure_news(headlines: list[dict], api_key: str, provider: str = "gemini") -> list[dict]:
    """Send headlines to an LLM, get structured signals back. Retries once on 429.

    provider: "gemini" (Google AI Studio key, starts with AIzaSy)
              "groq"   (console.groq.com key, starts with gsk_)
    """
    if not headlines:
        return []
    text = "\n".join(f"- [{h['source']}] {h['title']}" for h in headlines)
    prompt = PROMPT.replace("{headlines}", text)

    if provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"model": "llama-3.3-70b-versatile",
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.2}
    else:
        url = GEMINI_URL
        headers = {"x-goog-api-key": api_key}  # header, not URL: keeps key out of errors/logs
        payload = {"contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0.2}}

    import time
    for attempt in (1, 2):
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 429 and attempt == 1:
            time.sleep(20)  # free-tier per-minute limit: wait and retry once
            continue
        if resp.status_code != 200:
            raise RuntimeError(
                f"{provider} API error {resp.status_code}: "
                f"{resp.json().get('error', {}).get('message', 'unknown')[:200]}")
        break

    body = resp.json()
    if provider == "groq":
        raw = body["choices"][0]["message"]["content"]
    else:
        raw = body["candidates"][0]["content"]["parts"][0]["text"]
    return parse_signals(raw)


def parse_signals(raw: str) -> list[dict]:
    """Robustly parse the model's JSON (strips markdown fences if present)."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for d in data:
        if not isinstance(d, dict) or "headline" not in d:
            continue
        out.append({
            "headline": str(d.get("headline", ""))[:300],
            "source": str(d.get("source", "")),
            "commodity": str(d.get("commodity", "other")).lower(),
            "region": str(d.get("region", "EU")),
            "direction": str(d.get("direction", "neutral")).lower(),
            "impact": int(d.get("impact", 1)) if str(d.get("impact", 1)).isdigit() else 1,
            "rationale": str(d.get("rationale", "")),
        })
    return out


def get_market_drivers(api_key: str, provider: str = "gemini") -> dict:
    """Full pipeline. Returns {'fetched_at', 'n_headlines', 'signals'}."""
    headlines = fetch_headlines()
    signals = structure_news(headlines, api_key, provider=provider)
    return {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_headlines": len(headlines),
        "signals": signals,
    }
Headlines:
{headlines}
"""


def fetch_headlines(max_per_feed: int = 8) -> list[dict]:
    """Pull recent items from all feeds. Returns [{title, source, link, published}]."""
    items = []
    for source, url in FEEDS.items():
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:max_per_feed]:
                items.append({
                    "title": re.sub(r"\s+", " ", entry.get("title", "")).strip(),
                    "source": source,
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception:
            continue  # a dead feed should never break the dashboard
    # de-duplicate near-identical titles
    seen, unique = set(), []
    for it in items:
        key = it["title"].lower()[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(it)
    return unique


def structure_news(headlines: list[dict], api_key: str, provider: str = "gemini") -> list[dict]:
    """Send headlines to an LLM, get structured signals back. Retries once on 429.

    provider: "gemini" (Google AI Studio key, starts with AIzaSy)
              "groq"   (console.groq.com key, starts with gsk_)
    """
    if not headlines:
        return []
    text = "\n".join(f"- [{h['source']}] {h['title']}" for h in headlines)
    prompt = PROMPT.replace("{headlines}", text)

    if provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"model": "llama-3.3-70b-versatile",
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.2}
    else:
        url = GEMINI_URL
        headers = {"x-goog-api-key": api_key}  # header, not URL: keeps key out of errors/logs
        payload = {"contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0.2}}

    import time
    for attempt in (1, 2):
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 429 and attempt == 1:
            time.sleep(20)  # free-tier per-minute limit: wait and retry once
            continue
        if resp.status_code != 200:
            raise RuntimeError(
                f"{provider} API error {resp.status_code}: "
                f"{resp.json().get('error', {}).get('message', 'unknown')[:200]}")
        break

    body = resp.json()
    if provider == "groq":
        raw = body["choices"][0]["message"]["content"]
    else:
        raw = body["candidates"][0]["content"]["parts"][0]["text"]
    return parse_signals(raw)


def parse_signals(raw: str) -> list[dict]:
    """Robustly parse the model's JSON (strips markdown fences if present)."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for d in data:
        if not isinstance(d, dict) or "headline" not in d:
            continue
        out.append({
            "headline": str(d.get("headline", ""))[:300],
            "source": str(d.get("source", "")),
            "commodity": str(d.get("commodity", "other")).lower(),
            "region": str(d.get("region", "EU")),
            "direction": str(d.get("direction", "neutral")).lower(),
            "impact": int(d.get("impact", 1)) if str(d.get("impact", 1)).isdigit() else 1,
            "rationale": str(d.get("rationale", "")),
        })
    return out


def get_market_drivers(api_key: str, provider: str = "gemini") -> dict:
    """Full pipeline. Returns {'fetched_at', 'n_headlines', 'signals'}."""
    headlines = fetch_headlines()
    signals = structure_news(headlines, api_key, provider=provider)
    return {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_headlines": len(headlines),
        "signals": signals,
    }
