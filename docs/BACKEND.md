# Backend

## Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Web framework | FastAPI + Uvicorn | Typed request models, automatic `/docs` |
| Embeddings | `fastembed`, `BAAI/bge-small-en-v1.5` | ONNX, no PyTorch, small image, CPU-friendly; loaded lazily |
| Vector search | NumPy matmul on normalized vectors | Exact, no dependencies, fast to ~100k |
| LLM | Gemini REST via `httpx` | JSON mode, free tier; key sent in a header, not the URL |
| Persistence | JSON file, atomic write | Enough for a demo |
| Tests | pytest, fake embedder, mocked LLM | No downloads, runs in CI in seconds |

## Modules

```
app/
  main.py         create_app() factory: routes, lifespan seeding, static mount
  config.py       thresholds and environment variables
  normalize.py    normalize() and numbers()
  embeddings.py   lazy fastembed wrapper, L2-normalized float32 output
  store.py        items, vector matrix, exact index, search, coverage, diverse_sample, filter
  llm.py          ask_json() with retries and exponential backoff
  detector.py     check(): independent three-stage detector
  generator.py    plan_coverage() and generate()
```

## Data model

```json
{"id": "a1b2c3d4", "text": "Define the reflection of light.",
 "norm": "define the reflection of light", "tags": {"topic": "optics", "difficulty": "easy"}}
```

- `norm` powers O(1) exact lookup.
- Vectors live in a separate `float32` matrix aligned by index.
- `tags` is a free-form dict. Nothing in the detector reads it.

## Endpoints

| Method | Route | Body / params | Notes |
|--------|-------|---------------|-------|
| POST | `/api/check` | `{question}` | Read-only single classification |
| POST | `/api/check/batch` | `{questions: [...]}` | Batch classification for multiple questions |
| POST | `/api/generate` | `{topic, n}` (n 1 to 15) | Returns `plan`, `accepted`, `rejected`, `rounds`, optional `error` |
| POST | `/api/questions` | `{question, tags, force}` | Runs the detector; duplicates give `409`; `review` needs `force: true` |
| DELETE | `/api/questions/{id}` | none | Removes a question by ID |
| GET | `/api/questions` | `tag_key`, `tag_value` | Filter-only use of tags |
| GET | `/api/health` | none | Count and LLM availability |

Inputs are validated (length limits, `n` bounds). Interactive docs are at `/docs`.

## Detector flow

1. Exact normalized match returns `exact_duplicate`.
2. Search the top-3 neighbors. Best score below `REVIEW_T` returns `different`.
3. Best score at or above `AUTO_DUP_T` and no numeric mismatch returns `semantic_duplicate`.
4. Otherwise the judge sees all neighbors above `REVIEW_T`. It returns duplicate (with the matched neighbor), different, or ambiguous.
5. Judge unavailable: `semantic_duplicate` if score >= `FALLBACK_DUP_T` with no numeric mismatch, else `review`.

Every response includes `verdict`, `score`, `stage`, `match`, and `neighbors`.

## Generator flow

1. `plan_coverage`: LLM subtopics, ranked by `store.coverage` (lowest first).
2. Build context: top-15 relevant, top-3 per gap, 10 diverse (capped at 40 lines).
3. Prompt for 2x the needed questions with targets, existing questions, and recent rejections.
4. Vet each candidate with the detector. Accept only `different`; store it with `topic`, `source`, `subtopic` tags.
5. Repeat up to 3 rounds. If the LLM fails, stop and return an `error` with whatever was accepted.

## Reliability

- **Retries:** `ask_json` retries on network errors, 429/5xx and invalid JSON, with backoff (1s, 2s), then returns `None`.
- **Fallbacks:** callers treat `None` as "LLM unavailable" and degrade (`review`) instead of failing.
- **Thread safety:** the store uses an `RLock` around writes and reads.
- **Atomic persistence:** write to a temp file, then `os.replace`.
- **Secrets:** the API key goes in a request header and is never logged.

## Testing

`pytest` runs 23 tests in about a second with a deterministic bag-of-words embedder and a mocked LLM: normalization, exact/auto/judged/fallback paths, numeric guard, top-k judge prompt and match resolution, judge caching, tags having no effect, generator accept/reject and failure, LLM retry, and the API (409, filters, validation). A slow test (`RUN_SLOW=1`) runs the real model on the reflection corpus and asserts no false duplicates.

## Known limitations

- In-memory store: run a single worker.
- JSON rewrite on every add; all vectors re-embedded on startup.
- No authentication or rate limiting.
- Numeric guard covers digits only, not number words ("thirty").
