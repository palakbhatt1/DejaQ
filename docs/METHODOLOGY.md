# Methodology

How the project was scoped, designed, reviewed and improved.

## 1. Choosing the problem

The assignment offered two independent questions. Question 2 (duplicate prevention and detection for LLM-generated questions) was chosen over Question 1 (image catalog and composite generation) because:

- It could be built, tested and demoed completely within the deadline.
- Quality is measurable (precision, recall, confusion matrix), which supports a production-grade claim.
- It had fewer failure points for a recorded demo than a multi-model vision pipeline with a manual correction UI.

## 2. Requirements extracted from the prompt

| Requirement | How it is met |
|-------------|---------------|
| Bounded context when showing existing questions | Relevant + gap-targeted + diverse sample, capped at ~40 questions |
| Avoid same-meaning questions with different wording | Concept-level avoid instruction, rejected candidates fed back, coverage-targeted subtopics |
| Scales as the collection grows | Retrieval-based selection, swappable vector index, LLM only for the borderline band |
| Independent detector | Separate module and endpoint; shares only the store |
| Exact, paraphrase and equivalent-question detection | Normalize, embeddings, LLM judge |
| Tags are free-form and not used for dedup | Stored as a dict, filter-only, tested |
| Demonstrate exact, semantic and different cases | Reflection corpus in the UI and evaluation |
| Explain evaluation | `docs/EVALUATION.md` plus `eval/` scripts |

## 3. Design approach

**Layered defence.** Prevention reduces how many duplicates are produced. Detection guarantees that anything that slips through is still caught. Each layer is testable alone.

**Cheapest check first.** Hash lookup, then a vector dot product, then an LLM call. Most candidates are decided before the expensive step.

**LLM only where embeddings are weak.** Embeddings separate paraphrases from unrelated text well, but struggle with same-topic, different-task pairs. Only that band goes to the judge.

**Admit uncertainty.** Some pairs have no clear answer without context (Q8). The system returns `review` instead of forcing a label.

## 4. Design review and iteration

After the first version, the design was reviewed for failure modes. Changes and why:

| Finding | Change |
|---------|--------|
| A 0.90 auto-duplicate rule could mark Q7 a duplicate of Q6 (same words, different task) before the judge saw it | Auto-duplicate threshold raised to 0.96; everything in the band goes to the judge |
| Different given values can hide behind high similarity | Numeric guard: differing numbers force the judge |
| Nearest neighbor is not always the duplicated question | Judge sees the top-3 neighbors and says which one matched |
| 10 questions is a smoke test, not an evaluation | Labeled pair generator, precision/recall, threshold sweep |
| "Avoid these" does not tell the LLM what is missing | Coverage-aware generation: plan subtopics, target the least covered |
| LLM and UI robustness | Retries with backoff, judge cache, input validation, HTML escaping, 409 on duplicate add |
| No regression safety | 23 pytest tests and a GitHub Actions workflow |

## 5. Technology selection

| Decision | Chosen | Alternatives | Why |
|----------|--------|--------------|-----|
| Embeddings | bge-small via fastembed | Hosted embeddings, sentence-transformers | Local, no extra key, small image, no PyTorch |
| Index | NumPy | FAISS, Chroma, pgvector | Exact and trivial at this scale; interface allows swapping |
| LLM | Gemini Flash | Other hosted LLMs | Free tier, JSON mode, fast |
| UI | Vanilla HTML/JS | Streamlit, React | No Streamlit requirement; no build step needed |
| Hosting | Hugging Face Spaces (Docker) | Render, Railway, Fly.io | Free, public URL, matches existing HF presence |

## 6. Build order

1. Normalization and exact-match index.
2. Embeddings and the store with search.
3. Detector, verified against the reflection corpus before anything else depends on it.
4. Generator with retrieval, coverage planning and the reject-and-retry loop.
5. FastAPI routes and the UI.
6. Tests with a deterministic fake embedder and a mocked LLM.
7. Evaluation scripts: corpus, labeled pairs, sweep, latency.
8. Docker image, CI, deployment.

## 7. Testing and validation

- **Unit tests** cover every detector path (exact, auto, judged, fallback, numeric guard, caching), the generator, LLM retries and the API.
- **Corpus test** (real model, slow) asserts no false duplicates on Q5, Q6, Q7, Q9.
- **Labeled pairs** give precision/recall; the sweep sets thresholds from data rather than guesses.
- **Generation** is checked by re-running the detector over accepted output.

## 8. Limitations and honesty

- Results apply to the measured data; thresholds need re-tuning on a real question bank.
- Evaluation labels are LLM-produced and need a human spot-check.
- The judge is non-deterministic, so temperature is 0 and results are cached.
- Free-tier hosting is ephemeral and sleeps.

## 9. With more time

- Human-annotated evaluation set with agreement scores.
- Cross-encoder reranking before the judge.
- Persistent vectors (pgvector) and multi-worker support.
- Bulk import and check, and a review queue UI for `review` verdicts.
