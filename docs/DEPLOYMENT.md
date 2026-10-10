# Deployment

Target: **Hugging Face Spaces, Docker SDK.** Free, public URL for the demo video, same container as local.

## Prerequisites

- Hugging Face account and Git
- Gemini API key (Google AI Studio)
- Repo with `Dockerfile`, `requirements.txt`, and a `README.md` with the Spaces front-matter

## README front-matter (required)

```yaml
---
title: DejaQ
sdk: docker
app_port: 7860
---
```

`app_port` must match the port in the Dockerfile `CMD`.

## Test locally first

```bash
pytest                                  # logic, no downloads
docker build -t dejaq .
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key dejaq
curl localhost:7860/api/health          # {"status":"ok","questions":6,"llm":true}
```

Then open `http://localhost:7860`, check a question, generate, and list.

## Deploy steps

1. huggingface.co, **New Space**, SDK **Docker**, name `dejaq`.
2. Space **Settings → Variables and secrets**: add secret `GEMINI_API_KEY`.
3. Push the repo:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/dejaq
git push space main
```

4. Watch the **Logs** tab until the build finishes (a few minutes).
5. Open the Space URL and test.

If git asks for a password, use a Hugging Face access token with write permission.

## Dockerfile notes

- `python:3.11-slim` keeps the image small.
- The embedding model is downloaded at build time into `/app/models` (`FASTEMBED_CACHE_PATH`), so the first request does not wait on a download.
- The container runs as uid 1000 (`user`), which Spaces expects. `data/` is owned by that user so the store can write.
- Port `7860` is the Spaces default. Run a single worker (the store is in-memory).

## Environment variables

| Name | Where | Notes |
|------|-------|-------|
| `GEMINI_API_KEY` | Space secret | Never commit it |
| `GEMINI_MODEL` | Variable | Default `gemini-2.5-flash` |
| `AUTO_DUP_THRESHOLD`, `FALLBACK_DUP_THRESHOLD`, `REVIEW_THRESHOLD`, `JUDGE_TOP_K`, `LLM_RETRIES` | Variables | Set the values suggested by `eval/run_eval.py` |

## CI

`.github/workflows/ci.yml` runs `pytest` on every push and pull request. The tests use a fake embedder and a mocked LLM, so CI needs no secrets and no model download.

## Free tier behavior

- **Ephemeral disk:** the collection resets to `data/seed_questions.json` on restart or rebuild. Say so in the README.
- **Sleep:** the Space sleeps when idle. The first request after waking is slow, so open it once before recording.
- **Resources:** CPU only. The ONNX embedding model fits comfortably.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Build fails immediately | Missing or malformed README front-matter | Check the YAML block |
| "Application error" | Port mismatch | `app_port` and the `CMD` port must both be 7860 |
| Permission denied writing `data/` | Wrong user or ownership | Keep the `chown` and `USER user` lines |
| Generate returns an error | Missing or invalid key, or quota | Secret must be named exactly `GEMINI_API_KEY`; check `/api/health` |
| Everything borderline is `review` | No LLM key | Set the secret |
| Slow first request | Cold start | Wake the Space before the demo |
| Push rejected | Auth | Use a token with write access |

## Alternatives

| Platform | Notes |
|----------|-------|
| Render | Docker from GitHub; persistent disk on paid plans |
| Railway | Simple Docker deploys, usage-based billing |
| Fly.io | Persistent volumes, more setup |

For production, replace the JSON file with Postgres + pgvector and add authentication and rate limiting.

## Demo recording checklist (~2 min)

The seed has Q1, Q2, Q5, Q6, Q7 and Q9. Q3, Q4, Q8 and Q10 are not in it, so they behave as new candidates.

1. Wake the Space and send one request.
2. Check **Q10** "Define the reflection of light." → exact duplicate (stage `normalize`).
3. Check **Q4** "Explain the meaning of light reflection." → semantic duplicate (show the judge reason if it reached the LLM).
4. Check **"The angle of incidence is 45°. What is the angle of reflection?"** → different, with the `numeric guard` chip, which shows why a high-similarity pair is not merged.
5. Check **Q8** "What is reflection?" → review.
6. Generate 5 for "reflection of light" → show the coverage plan, accepted vs. rejected with scores.
7. Show the filter by tag and the confusion matrix from `run_eval`.
