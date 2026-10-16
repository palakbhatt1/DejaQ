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
    """Call Gemini in JSON mode. Retries across models with backoff; returns None on failure."""
    if not GEMINI_KEY:
        return None
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": temperature},
    }
    models = list(dict.fromkeys([GEMINI_MODEL, "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash"]))
    for model in models:
        for attempt in range(2):
            try:
                r = httpx.post(URL.format(model=model), headers={"x-goog-api-key": GEMINI_KEY},
                               json=body, timeout=30)
                if r.status_code in RETRYABLE:
                    raise httpx.HTTPError(f"retryable status {r.status_code}")
                r.raise_for_status()
                return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
                log.warning("LLM call for model %s failed (attempt %d/2): %s", model, attempt + 1, e)
                if attempt == 0:
                    time.sleep(0.5)
                else:
                    break
    return None
