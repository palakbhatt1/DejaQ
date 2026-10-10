"""Retrieval-guided, coverage-aware generation. Every candidate is vetted by the independent detector."""
import math

from . import llm
from .detector import check

PLAN = """List {k} distinct subtopics that exam questions on "{topic}" could cover (definitions, laws, types, numericals, applications, common misconceptions, comparisons).
Reply JSON: {{"subtopics":["..."]}}"""

GEN = """Write {n} NEW exam questions on: {topic}.

Spread the questions across these under-represented subtopics:
{targets}

Existing questions. Do NOT repeat them, rephrase them, or test the same concept or the same skill:
{covered}

Rejected earlier as too close (avoid these ideas): {rejected}

Vary the question type (definition, reasoning, numerical, application, comparison).
Reply JSON: {{"questions":[{{"question":"...","subtopic":"..."}}]}}"""

MAX_COVERED = 40


def plan_coverage(store, topic: str, k: int = 8) -> list[dict]:
    """Ask the LLM for subtopics, then rank them by how little the collection already covers them."""
    out = llm.ask_json(PLAN.format(k=k, topic=topic), temperature=0.3)
    subs = [s for s in (out or {}).get("subtopics", []) if isinstance(s, str)][:k]
    if not subs:
        return []
    cov = store.coverage([f"{topic}: {s}" for s in subs])
    return sorted(({"subtopic": s, "coverage": round(c, 3)} for s, c in zip(subs, cov)),
                  key=lambda x: x["coverage"])


def generate(store, topic: str, n: int = 5, rounds: int = 3) -> dict:
    plan = plan_coverage(store, topic)
    gaps = [p["subtopic"] for p in plan[:max(3, n)]]
    accepted, rejected, rejected_txt, error = [], [], [], None

    for r in range(rounds):
        need = n - len(accepted)
        if need <= 0:
            break
        near = [m["text"] for m, _ in store.search(topic, k=15)]
        for g in gaps:
            near += [m["text"] for m, _ in store.search(f"{topic}: {g}", k=3)]
        spread = [m["text"] for m in store.diverse_sample(10)]
        covered = list(dict.fromkeys(near + spread))[:MAX_COVERED]

        out = llm.ask_json(GEN.format(
            n=need * 2, topic=topic,
            targets="\n".join("- " + g for g in gaps) or "- any subtopic not yet covered",
            covered="\n".join("- " + t for t in covered) or "- (none yet)",
            rejected="; ".join(rejected_txt[-5:]) or "none"), temperature=0.8)
        if not out:
            error = "LLM unavailable or returned invalid output"
            break

        for item in out.get("questions", []):
            q = item.get("question") if isinstance(item, dict) else item
            if not isinstance(q, str) or not q.strip():
                continue
            res = check(store, q)  # also compares against questions accepted earlier in this batch
            if res["verdict"] == "different":
                sub = item.get("subtopic") if isinstance(item, dict) else None
                tags = {"topic": topic, "source": "generated", **({"subtopic": sub} if sub else {})}
                store.add(q, tags)
                accepted.append({"question": q, **res})
            else:
                rejected.append({"question": q, **res})
                rejected_txt.append(q)
            if len(accepted) >= n:
                break

    return {"plan": plan, "accepted": accepted, "rejected": rejected, "rounds": r + 1 if rounds else 0,
            **({"error": error} if error else {})}
