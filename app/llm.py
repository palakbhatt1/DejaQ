import json
import logging
import time

import httpx

from .config import GEMINI_KEY, GEMINI_MODEL, LLM_RETRIES

log = logging.getLogger("dejaq.llm")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE = {429, 500, 502, 503, 504}


def available() -> bool:
    return bool(GEMINI_KEY)


def ask_json(prompt: str, temperature: float = 0.2):
    """Call Gemini in JSON mode. Retries with exponential backoff; returns None on failure."""
    if not GEMINI_KEY:
        return None
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": temperature},
    }
    for attempt in range(LLM_RETRIES):
        try:
            r = httpx.post(URL.format(model=GEMINI_MODEL), headers={"x-goog-api-key": GEMINI_KEY},
                           json=body, timeout=40)
            if r.status_code in RETRYABLE:
                raise httpx.HTTPError(f"retryable status {r.status_code}")
            r.raise_for_status()
            return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            log.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, LLM_RETRIES, e)
            if attempt < LLM_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None
