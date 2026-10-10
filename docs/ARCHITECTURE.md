# Architecture

## 1. Problem and constraints

A question bank holds hundreds to thousands of questions. An LLM generates more for the same subject. Requirements:

- Minimize duplicates at generation time without exceeding the context window.
- Catch paraphrases, not just exact repeats.
- Independently detect duplicates if prevention fails.
- Tags are free-form key-value pairs and must not drive dedup. Only question content counts.

## 2. System overview

```
  topic, n ──► Generator (prevention)
               1. plan subtopics, rank by coverage
               2. build bounded context (relevant + gaps + diverse)
               3. LLM writes candidates
                         │
                         ▼
               Detector (independent)
               normalize → embed (top-3) → numeric guard → LLM judge
                         │
          accept ◄───────┴───────► reject (fed back as "avoid")
             │
             ▼
          Store: text, normalized text, vector, tags (filter-only)
```

The detector is also exposed on its own (`POST /api/check`) and shares no logic with the generator. The only shared component is the store.

## 3. Generation-time prevention

**Coverage-aware planning.** The LLM lists subtopics for the topic (laws, types, numericals, applications...). Each subtopic is embedded and scored by its maximum similarity to any stored question. The lowest-scoring subtopics are the gaps, and generation targets them. This turns "avoid duplicates" into "fill what is missing".

**Bounded context.** The prompt contains at most about 40 existing questions regardless of collection size:

- top 15 by relevance to the topic
- top 3 for each target subtopic
- 10 chosen by farthest-point sampling, so the LLM sees the overall spread of the collection

**Concept-level avoidance.** The prompt says not to repeat, rephrase, or test the same concept or skill, and asks for varied question types.

**Feedback loop.** Candidates rejected by the detector are fed into the next round as "avoid". Up to 3 rounds run; each requests 2x the needed candidates to absorb rejections.

**Within-batch duplicates.** Accepted questions go into the store immediately, so later candidates in the same batch are checked against them.

## 4. Independent duplicate detection

| Stage | Method | Cost | Output |
|-------|--------|------|--------|
| 1. Exact | Normalize (NFKC, lowercase, strip punctuation, collapse whitespace) then hash lookup | O(1) | `exact_duplicate` |
| 2. Embedding | `bge-small-en-v1.5`, cosine similarity to the top-3 neighbors | O(N) brute force | `different` (< 0.75) or `semantic_duplicate` (>= 0.96, no numeric mismatch) |
| 3. Judge | One LLM call sees the candidate plus every neighbor >= 0.75 and returns duplicate (with which one), different, or ambiguous | 1 LLM call | `semantic_duplicate`, `different`, `review` |

**Why the auto-duplicate threshold is high.** Small embedding models score short same-topic questions very high. "What is the angle of reflection?" and "The angle of incidence is 30°. What is the angle of reflection?" share most words but are different questions (definition vs. calculation). A low auto-duplicate threshold would throw away a good question without the judge ever seeing it. Only near-identical text is merged without the judge, and everything else in the band goes to it.

**Numeric guard.** If the numbers in the candidate and its nearest neighbor differ, the pair is never auto-merged, regardless of similarity. Different given values mean a different question.

**Top-k for the judge.** The nearest neighbor is not always the one the candidate duplicates. Showing the top neighbors in one call fixes that at the same LLM cost. The judge returns which neighbor matched.

**Ambiguous cases.** "What is reflection?" is broader than "What is reflection of light?" and depends on context. The judge may return `ambiguous`, which becomes `review` for a human. Generation rejects `review` candidates; the API lets a human add them with `force`.

**Degradation without an LLM.** Borderline cases return `review`, except pairs >= 0.90 with no numeric mismatch, which are marked duplicates as a conservative fallback.

**Judge caching.** Results are cached by (candidate, neighbors), so repeated checks cost nothing.

## 5. Tags and metadata

Tags are stored as a free-form dict per question. New keys and values need no schema change. The detector never reads them. `GET /api/questions?tag_key=&tag_value=` filters on them. A test asserts that different tags give the same verdict.

## 6. Scaling

| Collection size | Approach |
|-----------------|----------|
| up to ~100k | NumPy matrix multiply, exact search |
| 100k to millions | FAISS (HNSW/IVF) or pgvector behind the same `search()` interface |
| Embedding cost | One embedding per candidate; collection vectors computed once at insert |
| LLM cost | Only the borderline band; prompt size constant |
| Persistence | JSON for the demo; Postgres + pgvector for production |

## 7. Trade-offs

- **Small embedding model:** fast, CPU-only, small image. A larger model separates paraphrases better at higher latency and size.
- **Judge on the whole band:** more LLM calls than a single threshold, but fewer false duplicates. Duplicate precision is prioritized because a false duplicate discards a good question.
- **Coverage via embeddings:** a subtopic phrase vs. a question is a noisy similarity, so coverage ranking is a heuristic for targeting. The detector remains the safeguard.
- **LLM-generated eval labels:** cheap and fast, but need a human spot-check.
- **Single process:** the store is lock-protected but in-memory. Multi-worker needs a database.
