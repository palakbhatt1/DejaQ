"""Build a larger labeled pair set (eval/pairs.json) with Gemini. Needs GEMINI_API_KEY.

Labels are LLM-produced, so spot-check them by hand: delete bad pairs, set "reviewed": true on the ones you checked.
Run: python -m eval.generate_pairs
"""
import json
import random
from pathlib import Path

from app import llm

SUBJECTS = ["reflection and refraction of light", "photosynthesis", "Newton's laws of motion",
            "quadratic equations", "the French Revolution", "chemical bonding"]
PER_SUBJECT = 24

PROMPT = """Create {n} labeled question pairs for testing duplicate detection in an exam question bank on: "{subject}".
Each pair has "a" and "b" (exam questions) and a "label":
- "duplicate": same concept and same expected answer, different wording (paraphrase, reordering, synonyms, formal vs casual).
- "different": same topic but a genuinely different question (definition vs law vs numerical vs comparison vs application, or different given values).
Make about {half} duplicates and {half} different. At least half of the "different" pairs must be hard negatives (very similar wording but a different task or different numbers).
Reply JSON: {{"pairs":[{{"a":"...","b":"...","label":"duplicate"|"different","kind":"paraphrase"|"hard_negative"|"easy_negative"}}]}}"""


def main():
    pairs = []
    for subject in SUBJECTS:
        out = llm.ask_json(PROMPT.format(n=PER_SUBJECT, half=PER_SUBJECT // 2, subject=subject), temperature=0.8)
        if not out:
            print(f"skipped {subject}: no LLM output (check GEMINI_API_KEY)")
            continue
        for p in out.get("pairs", []):
            if p.get("a") and p.get("b") and p.get("label") in ("duplicate", "different"):
                pairs.append({"subject": subject, "a": p["a"], "b": p["b"], "label": p["label"],
                              "kind": p.get("kind", "unknown"), "reviewed": False})
    # formatting-only variants: exact duplicates after normalization
    dups = [p for p in pairs if p["label"] == "duplicate"]
    for p in random.sample(dups, min(10, len(dups))):
        pairs.append({"subject": p["subject"], "a": p["a"], "b": p["a"].upper().replace("?", " ? ").replace(".", ""),
                      "label": "duplicate", "kind": "format", "reviewed": True})
    Path(__file__).with_name("pairs.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
    print(f"wrote {len(pairs)} pairs -> eval/pairs.json (spot-check the labels before trusting results)")


if __name__ == "__main__":
    main()
