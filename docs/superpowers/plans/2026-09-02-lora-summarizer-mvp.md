# LoRA Summarizer MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a local Apple Silicon CLI experiment that compares an untouched 4-bit Qwen model with a QLoRA summarization adapter.

**Architecture:** A deterministic preparation layer produces MLX chat JSONL; shared inference helpers keep prompts and generation settings identical; thin CLIs cover baseline, training, evaluation, and ad-hoc summarization. Pure-Python tests cover all behavior that does not require downloading a multi-gigabyte model.

**Tech Stack:** Python 3, Apple MLX, `mlx-lm[train]`, PyYAML, pytest, Hugging Face model `mlx-community/Qwen3-4B-Instruct-2507-4bit`.

## Global Constraints

- Everything runs locally on Apple Silicon; do not add CUDA or NVIDIA-specific dependencies.
- Keep the project command-line based; do not add a web UI, Docker, database, cloud infrastructure, or authentication.
- Use a 4-bit quantized instruction model and QLoRA; do not fuse the adapter.
- Use the exact system instruction and `Summarize the following text:\n\n<SOURCE TEXT>` user prompt from the request.
- Store adapter weights under `adapters/summarizer` and comparisons under `outputs/`.
- Use the current MLX chat dataset and LoRA configuration format with prompt masking.
- Do not claim model inference or training works unless that operation is actually run successfully.
- Git commits are omitted because the session sandbox prohibits creating `.git` metadata.

---

### Task 1: Deterministic MLX chat data preparation

**Files:**
- Create: `lora-summarizer/scripts/common.py`
- Create: `lora-summarizer/scripts/prepare_data.py`
- Create: `lora-summarizer/tests/test_prepare_data.py`

**Interfaces:**
- Produces: `SYSTEM_PROMPT`, `user_prompt(text)`, `chat_record(text, summary)`, `extract_source_and_reference(record)`, `read_jsonl(path)`, `prepare(input_path, output_dir, seed, train_ratio, valid_ratio)`.

- [ ] **Step 1: Write failing tests** covering the exact three-message chat record, blank-field rejection with line number, deterministic output, non-overlapping splits, and printed counts. Test with seven temporary raw records and compare two runs byte-for-byte.
- [ ] **Step 2: Verify red** with `python3 -m pytest tests/test_prepare_data.py -q`; expect import failure because scripts do not exist.
- [ ] **Step 3: Implement the common prompt contract and preparation CLI.** Validate JSON objects and non-empty string fields before any writes; shuffle a local list with `random.Random(seed)`; allocate test and validation with at least one item when the input has at least three records; write UTF-8 JSONL named `train.jsonl`, `valid.jsonl`, and `test.jsonl`; expose `--input`, `--output-dir`, `--seed`, `--train-ratio`, and `--valid-ratio`.
- [ ] **Step 4: Verify green** with `python3 -m pytest tests/test_prepare_data.py -q`; expect all tests to pass.

### Task 2: Dependency-light evaluation metrics and reports

**Files:**
- Create: `lora-summarizer/scripts/metrics.py`
- Create: `lora-summarizer/tests/test_metrics.py`
- Create: `lora-summarizer/tests/test_evaluate.py`

**Interfaces:**
- Produces: `tokenize(text)`, `rouge1_f1(reference, prediction)`, `rouge_l_f1(reference, prediction)`, `score(reference, prediction)`, and `render_comparison(rows, base_average, lora_average)`.

- [ ] **Step 1: Write failing tests** for case-insensitive word tokenization, duplicate-aware unigram F1, LCS-based ROUGE-L F1, empty strings returning zero, perfect matches returning one, and Markdown containing SOURCE/REFERENCE/BASE MODEL/LORA MODEL sections plus both averages.
- [ ] **Step 2: Verify red** with `python3 -m pytest tests/test_metrics.py tests/test_evaluate.py -q`; expect missing-module failures.
- [ ] **Step 3: Implement metrics** using `re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())`, `collections.Counter` overlap, harmonic-mean F1, and a dynamic-programming LCS row.
- [ ] **Step 4: Implement `render_comparison`** as a pure function in `evaluate.py`, with Markdown headings per example and a final metrics table.
- [ ] **Step 5: Verify green** with the same pytest command; expect all tests to pass.

### Task 3: Shared model generation, baseline, evaluation, and summarize CLIs

**Files:**
- Modify: `lora-summarizer/scripts/common.py`
- Create: `lora-summarizer/scripts/baseline.py`
- Complete: `lora-summarizer/scripts/evaluate.py`
- Create: `lora-summarizer/scripts/summarize.py`
- Create: `lora-summarizer/tests/test_cli.py`

**Interfaces:**
- Produces: `load_generator(model_name, adapter_path=None)`, `generate_summary(generator, source, max_tokens, temperature)`, `run_dataset(...)`, and CLI parsers for all inference entry points.

- [ ] **Step 1: Write failing tests** with injected fake generators for identical prompt usage, output record keys (`source`, `reference`, `generated_summary`, `generation_seconds`), base versus adapter selection, `--text`/`--file` mutual exclusion, missing file failure, and comparison generation from fake outputs.
- [ ] **Step 2: Verify red** with `python3 -m pytest tests/test_cli.py tests/test_evaluate.py -q`; expect missing functions/modules.
- [ ] **Step 3: Implement lazy MLX imports.** `load_generator` calls `mlx_lm.load(model_name, adapter_path=...)`; `generate_summary` applies `tokenizer.apply_chat_template(..., add_generation_prompt=True, tokenize=False)` and calls `mlx_lm.generate` with consistent limits. Keep imports lazy so unit tests run before MLX is installed.
- [ ] **Step 4: Implement baseline and evaluation.** Both read MLX chat records through `extract_source_and_reference`; baseline writes `outputs/baseline.jsonl`; evaluation runs or reuses base results, writes `outputs/lora.jsonl`, calculates both ROUGE scores, writes `outputs/comparison.md`, and prints averages.
- [ ] **Step 5: Implement summarize CLI.** Require exactly one of `--text` and `--file`; default to `adapters/summarizer`, fail clearly when adapter config/weights are absent, and omit adapter loading with `--base`.
- [ ] **Step 6: Verify green** with `python3 -m pytest tests/test_cli.py tests/test_evaluate.py -q`; expect all tests to pass.

### Task 4: Current mlx-lm QLoRA configuration and training wrapper

**Files:**
- Create: `lora-summarizer/configs/lora_config.yaml`
- Create: `lora-summarizer/scripts/train.py`
- Create: `lora-summarizer/tests/test_train.py`

**Interfaces:**
- Produces: `load_config(path)`, `build_command(config_path, iters=None, adapter_path=None)`, and a CLI accepting `--config`, `--iters`, and `--adapter-path`.

- [ ] **Step 1: Write failing tests** asserting the command starts with the active Python executable plus `-m mlx_lm lora --config`, override flags are appended, and YAML contains model, `train: true`, `fine_tune_type: lora`, `data: data`, rank 8, batch size 1, 300 iterations, max sequence length 2048, periodic evaluation, `mask_prompt: true`, and `adapter_path: adapters/summarizer`.
- [ ] **Step 2: Verify red** with `python3 -m pytest tests/test_train.py -q`; expect missing files/functions.
- [ ] **Step 3: Implement YAML and wrapper.** Print the selected model, configuration path, iterations, and adapter directory; invoke `subprocess.run(command, check=True)` without shell parsing. Support smoke testing via `--iters 5 --adapter-path adapters/smoke`.
- [ ] **Step 4: Verify green** with the same pytest command; expect all tests to pass.

### Task 5: Sample dataset, environment, documentation, and staged verification

**Files:**
- Create: `lora-summarizer/data/sample_raw.jsonl`
- Create: `lora-summarizer/requirements.txt`
- Create: `lora-summarizer/README.md`
- Create: `lora-summarizer/.gitignore`
- Create: `lora-summarizer/outputs/.gitkeep`
- Create: `lora-summarizer/adapters/.gitkeep`

**Interfaces:**
- Consumes: all preceding CLIs.
- Produces: copy-pasteable beginner workflow and immediately preparable sample data.

- [ ] **Step 1: Create 30 original sample pairs**: six each for news, technical, business, academic, and meeting-style text. Every reference must preserve material names, numbers, conclusions, and relationships without adding facts.
- [ ] **Step 2: Add dependencies and ignores.** Use `mlx-lm[train]`, PyYAML, and pytest; ignore `.venv`, caches, generated output JSONL/Markdown, and adapter weights while retaining placeholder directories.
- [ ] **Step 3: Write the README** beginning with the exact virtual-environment commands and showing prepare, baseline, 5-iteration smoke training, full 300-iteration training, evaluation, `--text`, `--file`, and `--base` commands. Explain LoRA, QLoRA, MLX versus CUDA/PyTorch, adapter location, changing the centralized model/config, expanding the dataset, lowering batch size/sequence length, metric interpretation, and the sample-data limitation.
- [ ] **Step 4: Run the complete unit suite** with `.venv/bin/python -m pytest -q`; expect all tests to pass.
- [ ] **Step 5: Install and verify dependencies** with `python3 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`, and `.venv/bin/python -c "import mlx, mlx_lm, yaml; print('imports ok')"`; expect `imports ok`.
- [ ] **Step 6: Prepare and inspect data** with `.venv/bin/python scripts/prepare_data.py --input data/sample_raw.jsonl --output-dir data`, then parse all three files and assert 30 total records, valid role order, and no blank content.
- [ ] **Step 7: Run a feasible base-model smoke test** on one test record with a small token limit. Record success only if the model download and generation complete.
- [ ] **Step 8: Run a feasible QLoRA smoke test** with `.venv/bin/python scripts/train.py --iters 5 --adapter-path adapters/smoke`. Verify adapter artifacts and record success only if training completes.
- [ ] **Step 9: Re-run all unit tests** and review the exact README commands against the implemented parser help.
