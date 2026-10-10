---
title: DejaQ
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# DejaQ

Intelligent question generation with coverage-aware redundancy prevention and an independent 3-stage duplicate detector.

**Live demo:** `<space-url>`  
**Demo video:** `<video-link>`

## Overview

A question bank grows over time, and an LLM asked to "generate more questions" keeps repeating itself, often with different wording. This project handles that with two independent layers:

1. **Prevention (generation time).** A coverage-aware generator plans subtopics, finds the ones the collection covers least, and gives the LLM a bounded, retrieval-selected view of existing questions.
2. **Detection (safeguard).** Every candidate goes through a separate three-stage detector. It does not trust the generator.

Duplicate logic uses question content only. Tags are free-form key-value pairs used for filtering, never for dedup.

## Detector

| Stage | Method | Decides |
|-------|--------|---------|
| 1 | Normalized string match | Exact duplicates (case, punctuation, spacing, unicode) |
| 2 | Embedding cosine similarity vs. top-3 neighbors (`bge-small-en-v1.5`) | Clearly different (< 0.75). Near-identical (>= 0.96) only if no numeric mismatch |
| 3 | LLM judge sees all neighbors above 0.75 in one call | Paraphrase vs. same-topic-different-question vs. ambiguous |

Safety rules:
- **Numeric guard:** if the given numbers differ (e.g. "30°" vs. none), the pair is never auto-merged. It always goes to the judge.
- **Graceful degradation:** without an LLM key, borderline cases return `review` (or `semantic_duplicate` if >= 0.90 and no numeric mismatch).

Verdicts: `exact_duplicate`, `semantic_duplicate`, `different`, `review`.

## Generator

1. LLM lists subtopics for the topic. Each is scored by how well the collection already covers it (max similarity to any stored question).
2. The least-covered subtopics become generation targets.
3. Prompt context is a fixed-size set: top-15 relevant + top-3 per target subtopic + 10 farthest-point diverse questions.
4. Each candidate is vetted by the detector. Accepted ones are stored immediately (so in-batch repeats are caught). Rejected ones are fed back as "avoid" for the next round (up to 3).

## Tech stack

FastAPI · fastembed (ONNX, no PyTorch) · NumPy cosine index · Gemini (generation + judge) · vanilla HTML/JS UI · Docker on Hugging Face Spaces · pytest + GitHub Actions

## Project structure

```
app/
  main.py         routes, seeding, static UI (create_app factory)
  config.py       thresholds and env vars
  normalize.py    text normalization + numeric extraction
  embeddings.py   lazy-loaded embedding model
  store.py        questions, vectors, tags, coverage, persistence
  llm.py          Gemini client with retry/backoff
  detector.py     independent 3-stage detector
  generator.py    coverage-aware generation loop
static/index.html
data/seed_questions.json
eval/             reflection_corpus.json, generate_pairs.py, run_eval.py
tests/            unit + API tests (no model download needed)
docs/             ARCHITECTURE, BACKEND, FRONTEND, EVALUATION, METHODOLOGY, DEPLOYMENT
.github/workflows/ci.yml
Dockerfile, requirements*.txt, .env.example
```

## Run locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
uvicorn app.main:app --reload --port 7860
```

Open `http://localhost:7860`. Without a key the detector still works; generation needs the key.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                       # fast tests, fake embedder, no downloads
RUN_SLOW=1 pytest -m slow    # real model on the reflection corpus
```

## Evaluation

```bash
python -m eval.generate_pairs   # builds eval/pairs.json (~150 labeled pairs, needs GEMINI_API_KEY)
python -m eval.run_eval         # confusion matrix, precision/recall, threshold sweep, latency
```

See `docs/EVALUATION.md`. Labels in `pairs.json` are LLM-produced and should be spot-checked by hand.

## API

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/check` | Classify a single question (read-only) |
| POST | `/api/check/batch` | Classify a list of questions in parallel |
| POST | `/api/generate` | Plan, generate and vet new questions for a topic |
| POST | `/api/questions` | Add a question. Duplicates return `409`; `review` needs `"force": true` |
| DELETE | `/api/questions/{id}` | Remove a question from the collection |
| GET | `/api/questions` | List questions; optional `tag_key` and `tag_value` filters |
| GET | `/api/health` | Status, question count, and LLM readiness |

```bash
curl -X POST localhost:7860/api/check -H "Content-Type: application/json" \
  -d '{"question": "Define the reflection of light."}'
```

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `GEMINI_API_KEY` | none | Generation and the judge |
| `GEMINI_MODEL` | `gemini-2.5-flash` | |
| `AUTO_DUP_THRESHOLD` | `0.96` | At or above (and no numeric mismatch): duplicate without the LLM |
| `FALLBACK_DUP_THRESHOLD` | `0.92` | Duplicate threshold used only when the LLM is unavailable |
| `REVIEW_THRESHOLD` | `0.75` | Below: different. Between this and the auto threshold: judge |
| `JUDGE_TOP_K` | `3` | Neighbors shown to the judge |
| `LLM_RETRIES` | `3` | Attempts with exponential backoff |
| `DATA_PATH` | `data/store.json` | Persistence file |

Thresholds are starting points. Tune them with `eval/run_eval.py`.

## Known limitations

- Single-process store (thread-safe, but run one worker). Use Postgres + pgvector for multi-worker deployments.
- Free Hugging Face Spaces storage is ephemeral: the collection resets to the seed on restart.
- Judge labels and evaluation pairs are LLM-produced; results are only as good as the spot-check.

## Docs

[Architecture](docs/ARCHITECTURE.md) · [Backend](docs/BACKEND.md) · [Frontend](docs/FRONTEND.md) · [Evaluation](docs/EVALUATION.md) · [Methodology](docs/METHODOLOGY.md) · [Deployment](docs/DEPLOYMENT.md)
