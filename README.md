# DejaQ

Question generation with semantic deduplication.

DejaQ generates questions for a given topic while checking that new candidates do not repeat questions already in the collection. It combines coverage-aware generation with an independent duplicate-checking pipeline, so generation-time instructions are not the only line of defense.

## Features

- **Coverage-aware generation:** identifies under-represented subtopics and directs generation toward them.
- **Bounded retrieval context:** selects relevant and diverse examples without sending the entire question bank to the LLM.
- **Independent duplicate detection:** checks candidates using normalized text, semantic similarity, numeric safeguards, and an LLM judge for ambiguous cases.
- **Hard-negative handling:** avoids automatically merging questions that look similar but ask for different things, such as a definition versus a numerical problem.
- **Flexible tags:** supports arbitrary key-value tags for filtering; tags do not influence duplicate decisions.
- **Batch checking and in-batch deduplication:** checks multiple candidates and verifies accepted questions against previously accepted candidates.
- **Evaluation and tests:** includes a reflection-of-light benchmark, detector tests, API tests, and evaluation scripts.
- **API and web interface:** exposes a FastAPI API and a browser-based interface.

## How It Works

### Generation pipeline

1. The generator identifies topic subareas and estimates which areas are under-represented in the question bank.
2. It retrieves a bounded set of relevant and diverse existing questions.
3. The LLM generates candidates using the selected context and prior rejection feedback.
4. Each candidate passes through the independent duplicate detector before it can be accepted.

### Duplicate detection pipeline

1. **Exact match:** normalize Unicode, case, punctuation, and whitespace, then check for an exact normalized match.
2. **Semantic similarity:** encode questions with `BAAI/bge-small-en-v1.5` and compare normalized embeddings using cosine similarity.
3. **Numeric safeguard:** route questions with differing numerical details for additional review rather than automatically merging them based on embedding similarity alone.
4. **LLM judgment:** compare close neighbors and classify the candidate as a duplicate, different, or requiring review. If the LLM is unavailable, the system uses conservative fallback behavior.

The stages are intentionally separate from generation: a candidate can still be rejected even if the LLM was instructed to avoid existing questions.

## Tech Stack

- Python 3.11+
- FastAPI and Uvicorn
- fastembed with `BAAI/bge-small-en-v1.5` embeddings
- Gemini API for question generation and ambiguous-case judgments
- NumPy for similarity calculations
- JSON-based local persistence
- Pytest for automated tests
- Docker for containerized deployment

## Quickstart

### Requirements

- Python 3.11 or newer
- A Gemini API key for LLM-based generation and judgment (optional for basic duplicate checking)

### Install

```bash
git clone https://github.com/palakbhatt1/DejaQ.git
cd DejaQ
pip install -r requirements.txt
```

### Configure the LLM (optional)

Set `GEMINI_API_KEY` in your environment.

**PowerShell**
```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

**macOS/Linux**
```bash
export GEMINI_API_KEY="your-api-key"
```

### Run

```bash
uvicorn app.main:app --reload --port 7860
```

Open `http://localhost:7860` for the web interface or `http://localhost:7860/docs` for interactive API documentation.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/check` | Check one question against the collection without storing it. |
| POST | `/api/check/batch` | Check a batch of questions. |
| POST | `/api/generate` | Generate candidates for a topic and return vetted questions. |
| POST | `/api/questions` | Validate and store a question. |
| GET | `/api/questions` | List questions, with optional tag filtering. |
| DELETE | `/api/questions/{id}` | Delete a question by ID. |
| GET | `/api/health` | Return service and readiness information. |

Example request:

```bash
curl -X POST http://localhost:7860/api/check \
  -H "Content-Type: application/json" \
  -d '{"question":"What do you understand by the reflection of light?"}'
```

## Configuration

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Enables Gemini-based generation and judgment. |
| `GEMINI_MODEL` | Gemini model name; defaults to `gemini-2.5-flash`. |
| `AUTO_DUP_THRESHOLD` | Similarity threshold for automatic semantic-duplicate classification. |
| `FALLBACK_DUP_THRESHOLD` | Threshold used by the fallback detector when the LLM judge is unavailable. |
| `REVIEW_THRESHOLD` | Lower similarity boundary for routing a candidate to further judgment. |
| `JUDGE_TOP_K` | Maximum number of close neighbors passed to the judge. |
| `LLM_RETRIES` | Retry count for transient LLM/API failures. |
| `DATA_PATH` | Path to the JSON question store. |

See `app/config.py` for the effective defaults and implementation details.

## Tests and Evaluation

Run the standard test suite:

```bash
pip install -r requirements-dev.txt
pytest
```

Run the real-model test (downloads/uses the embedding model):

```bash
RUN_SLOW=1 pytest -m slow
```

Evaluate the reflection corpus:

```bash
python -m eval.run_eval --skip-latency
```

Generate additional labeled test pairs (requires `GEMINI_API_KEY`):

```bash
python -m eval.generate_pairs
```

The included reflection corpus covers exact duplicates, paraphrases, questions on the same topic that test different concepts, numerical questions, and ambiguous cases. Similarity thresholds are heuristics and should be evaluated on a representative labeled dataset before production use.

## Project Structure

```
DejaQ/
├── app/
│   ├── config.py       # Configuration and thresholds
│   ├── detector.py     # Independent duplicate detection
│   ├── embeddings.py   # Embedding model wrapper
│   ├── generator.py    # Coverage-aware generation workflow
│   ├── llm.py          # Gemini client and retries
│   ├── main.py         # FastAPI application and routes
│   ├── normalize.py    # Text normalization and numeric extraction
│   └── store.py        # Question storage and similarity operations
├── data/               # Seed questions
├── docs/               # Architecture, API, evaluation, and deployment notes
├── eval/               # Benchmark corpus and evaluation scripts
├── static/             # Web interface
├── tests/              # Unit and API tests
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt
```
