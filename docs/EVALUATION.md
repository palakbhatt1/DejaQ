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

Results (LLM judge: OFF - Graceful Degradation / Fallback Mode):

| ID | Expected | Predicted | Stage | Score | Correct |
|----|----------|-----------|-------|-------|---------|
| Q2 | semantic_duplicate | semantic_duplicate | embedding-fallback | 0.947 | Yes |
| Q3 | semantic_duplicate | review | no-llm | 0.908 | Review (Safe fallback) |
| Q4 | semantic_duplicate | semantic_duplicate | embedding-fallback | 0.946 | Yes |
| Q5 | different | review | no-llm | 0.900 | Review (Safe fallback) |
| Q6 | different | review | no-llm | 0.830 | Review (Safe fallback) |
| Q7 | different | review | no-llm | 0.853 | Review (Numeric Guard triggered) |
| Q8 | review | review | no-llm | 0.870 | Yes |
| Q9 | different | review | no-llm | 0.759 | Review (Safe fallback) |
| Q10 | exact_duplicate | exact_duplicate | normalize | 1.000 | Yes |

Confusion matrix (rows expected, columns predicted):

| | exact | semantic | different | review |
|---|---|---|---|---|
| **exact** | 1 | 0 | 0 | 0 |
| **semantic** | 0 | 2 | 0 | 1 |
| **different** | 0 | 0 | 0 | 4 |
| **review** | 0 | 0 | 0 | 1 |

*Key finding:* In no-LLM mode, **zero false duplicates** are created (100% precision on duplicates: 0 false positives). All borderline cases are safely directed to human `review` until the Gemini LLM judge is active. Q7 is the key hard case: the numeric guard prevents auto-merging with Q6.

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
| Errors by kind (paraphrase / hard_negative / easy_negative / format) | |

Precision on duplicates matters most: a false duplicate discards a good question, while a miss only lets a repeat through to the next safeguard or a human.

## 3. Threshold sweep

An embedding-only sweep (cosine >= t means duplicate) over the labeled pairs, printed by `run_eval`. It shows where embeddings stop separating duplicates from hard negatives and suggests:

- `REVIEW_THRESHOLD`: the highest t with recall >= 0.98 (below it, almost nothing is a duplicate, so skip the judge)
- `AUTO_DUP_THRESHOLD`: the lowest t with precision >= 0.99 (above it, merge without the judge)

| Suggested REVIEW_THRESHOLD | Suggested AUTO_DUP_THRESHOLD | Fraction of pairs sent to the judge |
|---|---|---|
| | | |

Update `.env` or Space variables with these values and re-run.

## 4. Performance

`run_eval` times brute-force `search` on random 384-d vectors (100 queries each). Also record embedding time per question and judge latency.

| Collection size | Median (ms) | p95 (ms) |
|-----------------|-------------|----------|
| 1,000 | | |
| 10,000 | | |
| 100,000 | | |

## 5. Generation quality

Generate 5 questions for 3 topics and record:

| Metric | Value |
|--------|-------|
| Candidates rejected per round | |
| Rounds needed to reach n | |
| Accepted questions flagged by an independent re-check against the full collection | |
| Subtopics covered (coverage plan before vs. after) | |

## 6. Automated tests

`pytest` (23 tests, fake embedder, mocked LLM) covers the logic paths and runs in CI. `RUN_SLOW=1 pytest -m slow` runs the real model on the corpus and asserts that Q5, Q6, Q7 and Q9 are never marked duplicates and Q10 is exact.

## 7. Limitations

- Pair labels are LLM-produced; quality depends on the manual spot-check.
- Thresholds tuned on this set can overfit it; re-tune on your real question bank.
- The judge is non-deterministic. Run the pair evaluation several times and report the spread, or keep temperature at 0 (the default here).
- Six subjects are not all domains; math word problems and multi-part questions may behave differently.

## 8. Next steps

1. Two human annotators on a 300-pair set, with inter-annotator agreement.
2. Cross-encoder reranking before the judge.
3. Measure judge agreement with human labels on the borderline band only.
4. Scale test with 10k to 100k real questions, with and without FAISS.
