"""Independent duplicate detector. Depends only on the store and the LLM client, never on the generator."""
from . import llm
from .config import AUTO_DUP_T, FALLBACK_DUP_T, REVIEW_T, TOP_K
from .normalize import normalize, numbers

JUDGE = """You are checking an exam question bank for duplicates.

New question:
{q}

Existing questions:
{cands}

Is the new question a DUPLICATE of any existing one? A duplicate tests the same concept and expects essentially the same answer, even if worded differently.
It is NOT a duplicate if it only shares a topic but asks for something different (a definition vs a law vs a calculation vs a comparison), or if it uses different given values.
Reply JSON: {{"label":"duplicate"|"different"|"ambiguous","match":<number of the duplicate existing question, or null>,"reason":"<one short line>"}}
Use "ambiguous" only if the answer depends on context that is not given."""

_cache: dict = {}


def _judge(q: str, cands: list[str]):
    key = (normalize(q), tuple(normalize(c) for c in cands))
    if key in _cache:
        return _cache[key]
    prompt = JUDGE.format(q=q, cands="\n".join(f"{i + 1}. {c}" for i, c in enumerate(cands)))
    out = llm.ask_json(prompt, temperature=0.0)
    if isinstance(out, dict) and out.get("label") in {"duplicate", "different", "ambiguous"}:
        _cache[key] = out
        return out
    return None


def _res(verdict, score, stage, match, neighbors, **extra):
    return {"verdict": verdict, "score": round(score, 4), "stage": stage, "match": match,
            "neighbors": neighbors, **extra}


def check(store, q: str) -> dict:
    # Stage 1: exact (normalized) match
    hit = store.exact(normalize(q))
    if hit:
        return _res("exact_duplicate", 1.0, "normalize", hit["text"], [])

    # Stage 2: embedding similarity against the top-k neighbors
    hits = store.search(q, k=TOP_K)
    if not hits:
        return _res("different", 0.0, "empty", None, [])
    neighbors = [{"text": m["text"], "score": round(s, 4)} for m, s in hits]
    best, s = hits[0]
    if s < REVIEW_T:
        return _res("different", s, "embedding", best["text"], neighbors)

    # Different given values (e.g. 30 degrees vs none) must never be auto-merged.
    num_guard = numbers(q) != numbers(best["text"])
    if s >= AUTO_DUP_T and not num_guard:
        return _res("semantic_duplicate", s, "embedding", best["text"], neighbors)

    # Stage 3: LLM judge sees every neighbor above REVIEW_T in one call
    close = [(m, sc) for m, sc in hits if sc >= REVIEW_T]
    j = _judge(q, [m["text"] for m, _ in close])
    if j is None:  # judge unavailable: degrade safely
        if s >= FALLBACK_DUP_T and not num_guard:
            return _res("semantic_duplicate", s, "embedding-fallback", best["text"], neighbors)
        return _res("review", s, "no-llm", best["text"], neighbors, numeric_guard=num_guard)

    reason = j.get("reason")
    if j["label"] == "duplicate":
        k = j.get("match")
        m, sc = close[k - 1] if isinstance(k, int) and 1 <= k <= len(close) else close[0]
        return _res("semantic_duplicate", sc, "llm", m["text"], neighbors, reason=reason, numeric_guard=num_guard)
    if j["label"] == "ambiguous":
        return _res("review", s, "llm", best["text"], neighbors, reason=reason, numeric_guard=num_guard)
    return _res("different", s, "llm", best["text"], neighbors, reason=reason, numeric_guard=num_guard)
