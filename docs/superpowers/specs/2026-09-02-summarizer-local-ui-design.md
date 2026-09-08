# Summarizer Local Test UI Design

## Goal

Add a deliberately small local interface for comparing the untouched Qwen base model with the selected LoRA adapter on arbitrary English text.

## User experience

The application has one page with a clear title, a short privacy note, a large source-text field, and one **Compare summaries** button. Successful requests display two equally prominent result panels: **Base model** and **Best LoRA adapter**. Each panel contains the generated summary and elapsed generation time.

The button is disabled while generation is running and the page shows a concise working state because two model passes may take several seconds. Blank input is rejected before inference. Failures appear as plain, actionable messages without stack traces. The layout stacks on narrow screens and uses no decorative images because this is a utilitarian local tool.

## Architecture

A lightweight local Python HTTP service will reuse `scripts.common.load_generator` and `scripts.common.generate_summary`. It will load the base model once and the adapter-backed model once, retain both in memory, and run them sequentially for a `/api/compare` request. The selected adapter is `adapters/best`; users' text stays on their Mac.

The frontend is a small static page served by the same Python process, avoiding a second runtime, Node dependency, database, authentication, cloud hosting, or cross-origin setup. HTML, CSS, and browser JavaScript live in a focused `ui/` directory. The Python entry point lives in `scripts/ui.py` and exposes pure validation/comparison functions for automated tests.

## Data flow

The browser sends `{ "text": "..." }` to `POST /api/compare`. The server validates that `text` is a non-empty string no longer than 20,000 characters, generates the base summary, then generates the adapter summary with identical token and temperature settings. It returns:

```json
{
  "base": {"summary": "...", "seconds": 1.23},
  "lora": {"summary": "...", "seconds": 1.31}
}
```

The client renders text safely through DOM text properties rather than injecting HTML. One failed generation returns an error response instead of presenting a partial comparison as complete.

## Testing and verification

Test-first Python tests will cover blank input, oversized input, identical generation settings, response structure, and static-file/API routing using fake generators so normal tests do not load the 4B model. JavaScript stays minimal and dependency-free. Final verification includes the complete existing test suite, the new server tests, a local HTTP request using the real model and adapter, and a successful page response.

## Scope

This is local-only and intentionally excludes file upload, result history, streaming tokens, model selection, editable settings, metrics, accounts, persistence, deployment, and public hosting. The existing command-line workflow remains unchanged.
