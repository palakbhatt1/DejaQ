"""Evaluation: reflection corpus confusion matrix, labeled-pair metrics, threshold sweep, search latency.

Run from the repo root:  python -m eval.run_eval [--pairs eval/pairs.json] [--skip-latency]
Set GEMINI_API_KEY to evaluate the full pipeline (with the LLM judge). Without it, borderline cases return `review`.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from app import llm
from app.detector import check
from app.embeddings import DIM, embed
from app.store import Store

HERE = Path(__file__).parent
LABELS = ["exact_duplicate", "semantic_duplicate", "different", "review"]
DUPS = {"exact_duplicate", "semantic_duplicate"}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def run_corpus():
    corpus = json.load(open(HERE / "reflection_corpus.json"))
    store, rows = Store(), []
    print("\n== Reflection corpus (each question checked against the ones before it) ==")
    print(f"{'id':<5}{'expected':<20}{'predicted':<20}{'stage':<18}{'score':<7}ok")
    for item in corpus:
        if item["expected"] == "original":
            store.add(item["question"])
            continue
        r = check(store, item["question"])
        ok = r["verdict"] == item["expected"]
        rows.append((item["expected"], r["verdict"]))
        print(f"{item['id']:<5}{item['expected']:<20}{r['verdict']:<20}{r['stage']:<18}{r['score']:<7.3f}{'Y' if ok else 'N'}")
        store.add(item["question"])
    m = {e: {p: 0 for p in LABELS} for e in LABELS}
    for e, p in rows:
        m[e][p] += 1
    print("\nConfusion matrix (rows = expected, columns = predicted)")
    print(" " * 20 + "".join(f"{p:>20}" for p in LABELS))
    for e in LABELS:
        print(f"{e:<20}" + "".join(f"{m[e][p]:>20}" for p in LABELS))
    acc = sum(e == p for e, p in rows) / len(rows)
    print(f"\nAccuracy: {acc:.0%} ({sum(e == p for e, p in rows)}/{len(rows)})")
    return {"confusion": m, "accuracy": acc}


def run_pairs(path: Path):
    pairs = json.load(open(path))
    print(f"\n== Labeled pairs: {len(pairs)} from {path.name} ==")
    tp = fp = fn = tn = review = 0
    mistakes, by_kind = [], {}
    for p in pairs:
        s = Store()
        s.add(p["a"])
        r = check(s, p["b"])
        gold, pred = p["label"] == "duplicate", r["verdict"] in DUPS
        review += r["verdict"] == "review"
        tp += gold and pred
        fp += (not gold) and pred
        fn += gold and not pred
        tn += (not gold) and not pred
        k = by_kind.setdefault(p.get("kind", "?"), [0, 0])
        k[0] += 1
        if gold != pred:
            k[1] += 1
            mistakes.append((p, r))
    precision, recall, f1 = prf(tp, fp, fn)
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  review={review} ({review / len(pairs):.0%}, counted as not-duplicate)")
    print(f"Duplicate precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
    print("Errors by pair kind:", {k: f"{v[1]}/{v[0]}" for k, v in by_kind.items()})
    for p, r in mistakes[:15]:
        print(f"  [{p['label']} -> {r['verdict']}/{r['stage']} {r['score']:.2f}] {p['a']!r} | {p['b']!r}")

    # embedding-only threshold sweep (no LLM): shows where AUTO_DUP and REVIEW thresholds should sit
    A, B = embed([p["a"] for p in pairs]), embed([p["b"] for p in pairs])
    cos = (A * B).sum(axis=1)
    gold = np.array([p["label"] == "duplicate" for p in pairs])
    print("\nEmbedding-only sweep (cosine >= t => duplicate)")
    print(f"{'t':<7}{'precision':<11}{'recall':<9}F1")
    sweep, rec98, prec99 = [], None, None
    for t in np.arange(0.60, 0.995, 0.02):
        pred = cos >= t
        ttp, tfp, tfn = int((pred & gold).sum()), int((pred & ~gold).sum()), int((~pred & gold).sum())
        pr, rc, f = prf(ttp, tfp, tfn)
        sweep.append({"t": round(float(t), 2), "precision": pr, "recall": rc, "f1": f})
        print(f"{t:<7.2f}{pr:<11.3f}{rc:<9.3f}{f:.3f}")
        if rc >= 0.98:
            rec98 = float(t)
        if prec99 is None and pr >= 0.99 and ttp:
            prec99 = float(t)
    print(f"\nSuggested REVIEW_THRESHOLD (highest t with recall >= 0.98): {rec98}")
    print(f"Suggested AUTO_DUP_THRESHOLD (lowest t with precision >= 0.99): {prec99}")
    return {"n": len(pairs), "precision": precision, "recall": recall, "f1": f1, "review_rate": review / len(pairs),
            "sweep": sweep, "suggested_review_t": rec98, "suggested_auto_dup_t": prec99}


def run_latency():
    print("\n== search() latency, brute-force cosine, 100 queries each ==")
    rng, out = np.random.default_rng(0), {}
    print(f"{'N':<10}{'median ms':<12}p95 ms")
    for n in (1_000, 10_000, 100_000):
        V = rng.standard_normal((n, DIM)).astype("float32")
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        ts = []
        for _ in range(100):
            q = rng.standard_normal(DIM).astype("float32")
            q /= np.linalg.norm(q)
            t0 = time.perf_counter()
            idx = np.argsort(-(V @ q))[:3]
            ts.append((time.perf_counter() - t0) * 1000)
        out[n] = {"median_ms": statistics.median(ts), "p95_ms": float(np.percentile(ts, 95))}
        print(f"{n:<10}{out[n]['median_ms']:<12.2f}{out[n]['p95_ms']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(HERE / "pairs.json"))
    ap.add_argument("--skip-latency", action="store_true")
    a = ap.parse_args()
    print(f"LLM judge: {'ON' if llm.available() else 'OFF (borderline cases return review)'}")
    results = {"llm_judge": llm.available(), "corpus": run_corpus()}
    if Path(a.pairs).exists():
        results["pairs"] = run_pairs(Path(a.pairs))
    else:
        print(f"\n(no {a.pairs}; run `python -m eval.generate_pairs` for the larger labeled set)")
    if not a.skip_latency:
        results["latency"] = run_latency()
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print("\nSaved eval/results.json")


if __name__ == "__main__":
    main()
