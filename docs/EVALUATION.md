# Evaluation

Three levels, from the assignment's smoke test to a measurable benchmark.

```bash
python -m eval.generate_pairs   # once; needs GEMINI_API_KEY; writes eval/pairs.json
python -m eval.run_eval         # writes eval/results.json
```

Set `GEMINI_API_KEY` to evaluate the full pipeline. Without it, the run reports the no-LLM behavior (borderline cases become `review`). The script prints whether the judge was on. Fill the tables below from its output. Do not report numbers that were not measured.

## 1. Reflection corpus (assignment ground truth)

Each question is checked against the questions before it, in order. Q1 seeds the store and is not scored.

| ID | Question | Expected |
|----|----------|----------|
| Q2 | Define the reflection of light. | Semantic duplicate |
| Q3 | What do you understand by the reflection of light? | Semantic duplicate |
| Q4 | Explain the meaning of light reflection. | Semantic duplicate |
| Q5 | State the laws of reflection of light. | Different |
| Q6 | What is the angle of reflection? | Different |
| Q7 | The angle of incidence is 30°. What is the angle of reflection? | Different |
| Q8 | What is reflection? | Review |
| Q9 | What is the difference between regular and diffuse reflection? | Different |
| Q10 | Define the reflection of light. | Exact duplicate |

Results (LLM judge: ON / OFF):

| ID | Expected | Predicted | Stage | Score | Correct |
|----|----------|-----------|-------|-------|---------|
| Q2 | semantic_duplicate | | | | |
| Q3 | semantic_duplicate | | | | |
| Q4 | semantic_duplicate | | | | |
| Q5 | different | | | | |
| Q6 | different | | | | |
| Q7 | different | | | | |
| Q8 | review | | | | |
| Q9 | different | | | | |
| Q10 | exact_duplicate | | | | |

Confusion matrix (rows expected, columns predicted):

| | exact | semantic | different | review |
|---|---|---|---|---|
| **exact** | | | | |
| **semantic** | | | | |
| **different** | | | | |
| **review** | | | | |

Q7 is the key hard case: it must not be a duplicate of Q6. The numeric guard forces it to the judge.

## 2. Labeled pair set

`eval/generate_pairs.py` asks Gemini for labeled pairs across six subjects (reflection and refraction, photosynthesis, Newton's laws, quadratic equations, the French Revolution, chemical bonding): paraphrases, easy negatives, and **hard negatives** (same topic, different task or different numbers), plus formatting-only variants. Each pair is checked as "store contains `a`, candidate is `b`".

**Spot-check the labels by hand** before trusting results: remove bad pairs and set `"reviewed": true` on the ones you verified.

| Metric | Value |
|--------|-------|
| Pairs (reviewed / total) | |
| Duplicate precision | |
| Duplicate recall | |
| F1 | |
| Review rate | |
| Exact match recall | |
| Semantic duplicate recall | |
| Hard negative separation | |
| Numeric guard trigger rate | |

Run `python -m eval.run_eval --pairs eval/pairs.json` to compute these.

## 3. Threshold sweep

The script tests `REVIEW_THRESHOLD` in [0.65, 0.85] and `AUTO_DUP_THRESHOLD` in [0.90, 0.98] on the labeled pair set and prints:

- Precision vs recall curve for duplicate detection.
- Optimal thresholds maximizing F1 while keeping precision $\ge$ 0.95.
- The review rate across threshold pairs.

Report the best threshold pair here and update `app/config.py` defaults if they differ from the starting points:

| Parameter | Default | Tuned | Why |
|-----------|---------|-------|-----|
| `AUTO_DUP_THRESHOLD` | 0.96 | | |
| `FALLBACK_DUP_THRESHOLD` | 0.92 | | |
| `REVIEW_THRESHOLD` | 0.75 | | |

## 4. Latency and resource profile

Measure on the evaluation pair set:

| Component | Mean latency | p95 latency | Notes |
|-----------|--------------|-------------|-------|
| Hash lookup (exact) | < 1 ms | < 1 ms | In-memory dict |
| Vector search (top-3) | | | Brute-force dot product on $N$ items |
| LLM judge call | | | Gemini Flash REST call; cache hit is < 1 ms |
| End-to-end check (worst case) | | | Hash + embed + judge |

Vector search scale test:

| Store size | Search latency (ms) | Notes |
|------------|---------------------|-------|
| 100 | | NumPy dot product |
| 1,000 | | |
| 10,000 | | |
| 100,000 | | |

Run `python -m eval.run_eval --skip-latency` to skip the search benchmark if timing is not needed.
