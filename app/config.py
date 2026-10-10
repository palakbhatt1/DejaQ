import os


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# Detector thresholds (cosine similarity). Tune with `python -m eval.run_eval`.
AUTO_DUP_T = _f("AUTO_DUP_THRESHOLD", 0.96)          # >= : duplicate without LLM (if no numeric mismatch)
FALLBACK_DUP_T = _f("FALLBACK_DUP_THRESHOLD", 0.92)  # used only when the LLM judge is unavailable
REVIEW_T = _f("REVIEW_THRESHOLD", 0.75)              # < : different; [REVIEW_T, AUTO_DUP_T): LLM judge
TOP_K = int(os.getenv("JUDGE_TOP_K", 3))             # neighbors shown to the judge

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_RETRIES = int(os.getenv("LLM_RETRIES", 3))
DATA_PATH = os.getenv("DATA_PATH", "data/store.json")
