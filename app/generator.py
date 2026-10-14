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


def _fallback_candidates(topic: str, gaps: list[str]) -> list[dict]:
    t = topic.lower()
    if "reflect" in t or "light" in t or "mirror" in t or "optic" in t:
        return [
            {"question": "Explain why the image formed by a plane mirror is always laterally inverted, and contrast this with image formation in spherical mirrors.", "subtopic": "Image Formation & Mirrors"},
            {"question": "A light ray strikes a reflective surface at an angle of incidence of 50°. Calculate the angle between the incident ray and the reflected ray.", "subtopic": "Numerical Problems"},
            {"question": "How do retroreflectors on bicycles and road signs utilize multiple reflections to direct light back toward vehicle headlights?", "subtopic": "Applications & Optical Devices"},
            {"question": "Why does diffuse reflection from rough surfaces obey the laws of reflection locally, even though reflected rays scatter in all directions?", "subtopic": "Laws & Principles"},
            {"question": "Compare the characteristics of real images formed by concave mirrors with virtual images formed by convex mirrors under varying object distances.", "subtopic": "Spherical Mirrors"},
            {"question": "What is the physical difference between specular reflection and diffuse reflection, and what determines which occurs on a given surface?", "subtopic": "Types of Reflection"},
        ]
    if "photo" in t or "plant" in t or "bio" in t:
        return [
            {"question": "Compare the primary carbon fixation pathways and enzymatic adaptations of C3, C4, and CAM plants under drought conditions.", "subtopic": "Carbon Fixation Pathways"},
            {"question": "Explain the role of chlorophyll a and accessory pigments in the light-harvesting complex during the light-dependent reactions of photosynthesis.", "subtopic": "Light Reactions & Pigments"},
            {"question": "How does increasing atmospheric carbon dioxide concentration impact the photosynthetic efficiency of agricultural crops experiencing high temperatures?", "subtopic": "Environmental Factors"},
            {"question": "Describe the biochemical mechanism of non-cyclic photophosphorylation and identify the terminal electron acceptor in the thylakoid membrane.", "subtopic": "Electron Transport & ATP Synthesis"},
        ]
    return [
        {"question": f"Define the fundamental scientific principles governing {topic} and explain their primary conceptual significance.", "subtopic": gaps[0] if gaps else "Definitions & Principles"},
        {"question": f"How do theoretical laws of {topic} apply to modern real-world technological and industrial systems?", "subtopic": gaps[1] if len(gaps) > 1 else "Real-World Applications"},
        {"question": f"Contrast the primary mechanisms of {topic} with related alternative processes under varying standard conditions.", "subtopic": gaps[2] if len(gaps) > 2 else "Comparative Analysis"},
        {"question": f"What is a common misconception students encounter regarding {topic}, and how does empirical evidence clarify it?", "subtopic": "Misconceptions & Conceptual Reasoning"},
        {"question": f"In an experiment analyzing {topic}, what are the key dependent and independent variables required to measure accuracy?", "subtopic": "Experimental & Analytical Methods"},
    ]


def plan_coverage(store, topic: str, k: int = 8, allow_fallback: bool = False) -> list[dict]:
    """Ask the LLM for subtopics, then rank them by how little the collection already covers them."""
    out = llm.ask_json(PLAN.format(k=k, topic=topic), temperature=0.3)
    subs = [s for s in (out or {}).get("subtopics", []) if isinstance(s, str)][:k]
    if not subs:
        if allow_fallback:
            subs = ["Definitions & Principles", "Laws & Types", "Numerical Problems", "Applications & Devices", "Common Misconceptions"]
        else:
            return []
    cov = store.coverage([f"{topic}: {s}" for s in subs])
    return sorted(({"subtopic": s, "coverage": round(c, 3)} for s, c in zip(subs, cov)),
                  key=lambda x: x["coverage"])


def generate(store, topic: str, n: int = 5, rounds: int = 3, allow_fallback: bool = False) -> dict:
    plan = plan_coverage(store, topic, allow_fallback=allow_fallback)
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
            if allow_fallback and not accepted:
                candidates = _fallback_candidates(topic, gaps)
            else:
                error = "LLM unavailable or returned invalid output"
                break
        else:
            candidates = out.get("questions", []) if isinstance(out, dict) else []

        for item in candidates:
            q = item.get("question") if isinstance(item, dict) else item
            if not isinstance(q, str) or not q.strip():
                continue
            res = check(store, q)  # also compares against questions accepted earlier in this batch
            if res["verdict"] == "different" or (res["verdict"] == "review" and res.get("stage") == "no-llm"):
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
