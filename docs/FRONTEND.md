# Frontend

## Overview

A single static page (`static/index.html`) served by FastAPI. Vanilla HTML, CSS and JavaScript: no framework, no build step, no Node toolchain.

## Why no framework

- Four interactions: check, add, generate, browse.
- A build step would add a Docker stage and config for no user-visible benefit.
- One file keeps the image small and the code reviewable.

If the UI grows (auth, multi-subject views, inline editing), move to React or Svelte with Vite and keep the same `/api/*` contract.

## Sections

| Section | Behavior |
|---------|----------|
| Check a question | Calls `/api/check`. Shows verdict, score, stage, matched question, judge reason, and the top neighbors with scores. Offers an **Add to collection** button for `different` (and a human-override add for `review`) |
| Generate | Calls `/api/generate`. Shows the coverage plan table, then accepted and rejected candidates with the same detail |
| Collection | Lists questions and tags. Filter by tag key and optional value |

## Verdict display

Each result shows the verdict, a score chip, a stage chip (`normalize`, `embedding`, `embedding-fallback`, `llm`, `no-llm`, `empty`), a `numeric guard` chip when it forced the judge, the matched question, the judge's one-line reason, and a collapsible neighbor list. The goal is a transparent, explainable decision rather than a black box.

## Design

Clean and minimal: system font, black and white, thin borders, clear hierarchy. Supports dark mode via `prefers-color-scheme`. Single column, max width 880px, works on mobile.

## Safety and robustness

- All question text, tags and server strings are HTML-escaped before rendering, so user-supplied questions cannot inject markup.
- Buttons disable during requests and show a loading message.
- Failed requests show an error row instead of failing silently.
- A rejected add (`409`) shows the verdict that blocked it.

## API contract used

| Action | Request | Fields used |
|--------|---------|-------------|
| Check | `POST /api/check` | `verdict, score, stage, match, reason, neighbors, numeric_guard` |
| Add | `POST /api/questions` | `question, tags, force`; handles `409` |
| Generate | `POST /api/generate` | `plan[], accepted[], rejected[], error` |
| List | `GET /api/questions?tag_key&tag_value` | `id, question, tags` |

## Known limitations

- No pagination in the collection list (fine for hundreds).
- No auth; anyone with the URL can add or generate.
- Tag filter is exact-match on one key.

## Possible improvements

- Highlight the differing words between a candidate and its match.
- Bulk check (paste or upload a list of questions).
- Pagination and search in the collection.
