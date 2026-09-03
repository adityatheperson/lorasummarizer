# Summarizer Local Test UI Design

## Goal

Add a deliberately small local interface for comparing the untouched Qwen base model with the selected LoRA adapter on arbitrary English text.

## Experience and architecture

The single page has a source-text field, one **Compare summaries** button, and equal Base Model and Best LoRA Adapter panels showing summary and elapsed time. It disables submission during inference, rejects blank input, renders errors plainly, and stacks on narrow screens.

A standard-library Python HTTP service serves static HTML/CSS/JavaScript and reuses `scripts.common` for inference. It loads the base and `adapters/best` models once, runs them sequentially with identical settings, and returns JSON from `POST /api/compare`. Text remains local; there is no upload, persistence, authentication, Node runtime, database, or hosting.

## Safety and verification

Input must be a non-empty string no longer than 20,000 characters. Browser output uses `textContent`, not HTML injection. Tests use fake generators to cover validation, comparison structure, shared settings, API routing, and static serving without loading MLX. Final verification includes all tests and a real local comparison request.

