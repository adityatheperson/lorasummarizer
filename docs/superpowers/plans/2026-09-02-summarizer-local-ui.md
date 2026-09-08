# Summarizer Local UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-page local base-versus-LoRA testing interface.

**Architecture:** A standard-library Python server owns lazy model loading, comparison, JSON routing, and static serving. A dependency-free page submits text and safely renders both results.

**Tech Stack:** Python 3, `http.server`, HTML, CSS, browser JavaScript, existing MLX/`mlx-lm` stack, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-summarizer-local-ui-design.md`

## Global Constraints

- Local-only; text never leaves the Mac except the existing Hugging Face model download.
- Use `adapters/best` and the existing default base model.
- Generate base and LoRA summaries with identical limits and temperature.
- No Node dependency, web framework, database, authentication, persistence, upload, or hosting.
- Reject blank text and text longer than 20,000 characters.
- Render model output with `textContent` rather than `innerHTML`.

---

### Task 1: Comparison service and HTTP API

**Files:**
- Create: `scripts/ui.py`
- Create: `tests/test_ui.py`

**Interfaces:**
- Produces: `validate_text(value) -> str`, `ComparisonService.compare(text) -> dict`, `create_server(host, port, service, ui_dir) -> ThreadingHTTPServer`.

- [ ] Write failing tests asserting blank/non-string/oversized rejection; base and LoRA calls receive identical `max_tokens=180` and `temperature=0.0`; response keys are `base` and `lora` with summary and nonnegative seconds; `/api/compare` accepts valid JSON and returns 400 for invalid input; `/` serves the page.
- [ ] Run `.venv/bin/python -m pytest tests/test_ui.py -q` and confirm the missing module causes failure.
- [ ] Implement `ComparisonService` with injected loader/generator callables, lazy cached base and adapter generators, sequential timing, and JSON-safe results. Implement a `BaseHTTPRequestHandler` subclass inside `create_server`, bounded request reads, JSON errors, static routing restricted to `index.html`, `styles.css`, and `app.js`, and a CLI defaulting to `127.0.0.1:8000`.
- [ ] Run `.venv/bin/python -m pytest tests/test_ui.py -q` and confirm all UI service tests pass.

### Task 2: One-page interface and documentation

**Files:**
- Create: `ui/index.html`
- Create: `ui/styles.css`
- Create: `ui/app.js`
- Modify: `README.md`

**Interfaces:**
- Consumes: `POST /api/compare` JSON contract.
- Produces: accessible local comparison page and launch instructions.

- [ ] Create semantic HTML with a labeled textarea, character counter, compare button, status region, error region, and two result sections containing summary and time elements.
- [ ] Create a restrained responsive stylesheet with readable typography, clear focus states, a two-column result grid above 760px, and stacked panels below it.
- [ ] Create JavaScript that enforces the 20,000-character limit, disables the button while awaiting `fetch('/api/compare')`, parses error JSON, writes every dynamic value through `textContent`, and restores controls in `finally`.
- [ ] Add README commands: `python scripts/ui.py`, open `http://127.0.0.1:8000`, paste text, click Compare summaries, and stop with Control-C.
- [ ] Run `.venv/bin/python -m pytest -q`, `.venv/bin/python -m compileall -q scripts tests`, and a static scan confirming `app.js` contains `textContent` and no `innerHTML`.

### Task 3: Real local smoke test and commit

**Files:**
- Verify: `scripts/ui.py`, `ui/*`, `README.md`

**Interfaces:**
- Consumes: selected adapter and cached base model.
- Produces: verified local page and comparison API.

- [ ] Start `.venv/bin/python scripts/ui.py --port 8765` outside the sandbox so MLX can access Metal.
- [ ] Request `/`, `/styles.css`, and `/app.js`; require HTTP 200 and recognizable content.
- [ ] POST a short factual passage to `/api/compare`; require both summaries and timing values.
- [ ] Stop the server, rerun the full tests, inspect `git diff`, and commit the verified UI.
