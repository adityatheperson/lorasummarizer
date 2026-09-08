# LoRA Summarizer MVP Design

## Goal

Build a minimal, reproducible, command-line project for Apple Silicon that demonstrates whether QLoRA fine-tuning improves English summarization over the untouched base model.

## Model and platform

The default model is `mlx-community/Qwen3-4B-Instruct-2507-4bit`, a quantized text-generation model sized comfortably for a MacBook with 32 GB unified memory. The project uses Python 3, Apple MLX, `mlx-lm`, and Hugging Face model distribution. It has no CUDA, NVIDIA, web UI, Docker, database, cloud infrastructure, or authentication dependency.

The model identifier is centralized so users can replace it consistently. Training uses the current `mlx_lm` LoRA command and the quantized model, which makes the run QLoRA. Adapter weights remain separate under `adapters/summarizer`; the project does not fuse them into the base model.

## Project structure and responsibilities

The repository will contain a self-contained `lora-summarizer/` directory:

- `README.md`: Apple Silicon setup, copy-paste workflow, concepts, tuning guidance, and test-status disclosures.
- `requirements.txt`: minimal runtime and test dependencies.
- `data/sample_raw.jsonl`: approximately 30 synthetic `{text, summary}` pairs spanning news, technical, business, academic, and meeting content.
- `data/{train,valid,test}.jsonl`: deterministic MLX chat-format splits generated from the raw sample.
- `scripts/prepare_data.py`: input validation, deterministic shuffle, split calculation, and MLX chat serialization.
- `scripts/common.py`: canonical system/user prompts, JSONL loading, chat-template application, model loading, and generation helpers shared across CLIs.
- `scripts/metrics.py`: dependency-light ROUGE-1 and ROUGE-L F1 metrics.
- `scripts/baseline.py`: base-model inference over every test example and `outputs/baseline.jsonl` creation.
- `scripts/train.py`: thin wrapper that prints configuration and invokes the supported `python -m mlx_lm lora --config ...` command.
- `scripts/evaluate.py`: comparable base and adapter inference, per-example outputs, aggregate metrics, and Markdown comparison report.
- `scripts/summarize.py`: inference from `--text` or `--file`, using the adapter by default and the untouched model with `--base`.
- `configs/lora_config.yaml`: conservative 32 GB defaults: rank 8, batch size 1, 2048-token sequences, learning rate in the supported `1e-5` to `1e-4` range, 300 iterations, periodic validation, prompt masking, and adapter output under `adapters/summarizer`.
- `tests/`: focused automated tests for deterministic preparation, invalid input, prompt construction, ROUGE calculations, report creation, and CLI validation without downloading a model.
- `outputs/` and `adapters/`: generated result and adapter locations, retained with placeholder files until runs populate them.

## Data flow

`prepare_data.py` reads one JSON object per line with non-empty string `text` and `summary` fields. It validates the entire input before writing outputs, shuffles with a documented fixed seed, and creates non-overlapping train, validation, and test splits. Each output record uses the current MLX chat dataset shape: a `messages` array with system, user, and assistant roles. The final assistant message is the completion, enabling `mask_prompt: true` to train loss primarily on the target summary.

Baseline inference and adapter inference both use the same test set, chat template, generation parameters, and helper code. Baseline results are stored in `outputs/baseline.jsonl`; tuned results are stored in `outputs/lora.jsonl`. Each record contains the source, reference, generated summary, and elapsed generation time.

## Training and evaluation

`train.py` loads and displays the YAML settings, allows a small iteration override for smoke testing, and invokes the installed `mlx_lm` CLI rather than duplicating its training internals. Full training defaults to approximately 300 iterations. A smoke-test option runs 5–10 iterations into a separate adapter directory so it does not overwrite the intended full adapter.

`evaluate.py` can reuse a compatible existing baseline file or regenerate the base outputs, then loads the LoRA adapter and generates tuned outputs. It calculates ROUGE-1 F1 and ROUGE-L F1 for every example and averages them separately for the base and LoRA models. `outputs/comparison.md` shows the source, reference, base output, LoRA output, per-example scores, and averages, making improvement or regression observable rather than assuming training helped.

## Error handling

CLI failures use actionable messages for malformed JSONL, missing/blank fields, nonexistent files, mutually exclusive input options, missing adapter weights, model-loading failures, and subprocess failures. File writes create required parent directories and use UTF-8 JSONL. Generation outputs are written incrementally or atomically where practical so a failed run does not masquerade as a complete evaluation.

## Verification strategy

Implementation follows test-first development for locally testable behavior. Tests avoid model downloads through narrow dependency boundaries. Verification proceeds in this order:

1. Run automated tests.
2. Create a virtual environment and install the pinned/minimally constrained dependencies.
3. Verify `mlx`, `mlx_lm`, and project imports.
4. Run data preparation and validate the generated split counts and JSONL structure.
5. Run a tiny base-model generation if the model download, network, disk, and execution time are feasible.
6. Run 5–10 QLoRA iterations and verify adapter artifacts if feasible.
7. Report each verification truthfully, distinguishing completed checks from commands the user still needs to run.

The synthetic dataset is only pipeline test data. The README will state plainly that meaningful experiments require hundreds—and preferably thousands—of carefully reviewed source-summary pairs. Users replacing it with real data should modify or replace `data/sample_raw.jsonl`, preserving the `{text, summary}` schema, and rerun preparation.

## Scope decisions

The MVP intentionally avoids experiment tracking services, model fusion, distributed training, GPU portability, rich metric libraries, and graphical interfaces. It favors a small number of explicit scripts and one shared utility module over a framework or package hierarchy.
