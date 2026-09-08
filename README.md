# Local LoRA Summarizer MVP for Apple Silicon

This command-line project tests whether a small QLoRA adapter improves English summarization over the untouched base model. It defaults to the 4-bit `mlx-community/Qwen3-4B-Instruct-2507-4bit`, which is practical on a 32 GB Apple Silicon Mac.

The included 30 examples are synthetic pipeline-test data—not enough to establish production quality. A meaningful experiment should use hundreds, and preferably thousands, of carefully reviewed source-summary pairs.

## Setup

Run these commands from this directory on an Apple Silicon Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MLX uses Apple's unified-memory and Metal stack directly. This project therefore does not need CUDA, an NVIDIA GPU, or a CUDA-oriented PyTorch training setup.

## 1. Prepare the dataset

```bash
python scripts/prepare_data.py \
  --input data/sample_raw.jsonl \
  --output-dir data
```

This validates each `{\"text\": \"...\", \"summary\": \"...\"}` record, shuffles with seed 42, and writes MLX chat-format `train.jsonl`, `valid.jsonl`, and `test.jsonl`.

## 2. Run the untouched baseline

The first run downloads roughly 2–3 GB of model files from Hugging Face.

```bash
python scripts/baseline.py
```

Results go to `outputs/baseline.jsonl`. For a one-example smoke test:

```bash
python scripts/baseline.py --limit 1 --max-tokens 64 --output outputs/baseline-smoke.jsonl
```

## 3. Train the QLoRA adapter

First run five iterations without touching the full adapter directory:

```bash
python scripts/train.py --iters 5 --adapter-path adapters/smoke
```

Then run the 300-iteration MVP configuration:

```bash
python scripts/train.py
```

Training and validation losses are printed by `mlx-lm`. The unfused adapter weights are stored in `adapters/summarizer`.

## 4. Evaluate base versus LoRA

```bash
python scripts/evaluate.py
```

This reuses `outputs/baseline.jsonl` when it matches the test set, generates adapter summaries in `outputs/lora.jsonl`, and creates `outputs/comparison.md`. To force fresh base results:

```bash
python scripts/evaluate.py --regenerate-baseline
```

The report contains every source, reference, base output, and LoRA output plus average ROUGE-1 F1 and ROUGE-L F1. Higher values indicate more word and sequence overlap with the references, but you should also read the outputs for factual accuracy, omissions, and unsupported claims. Improvement is evidence from the held-out comparison, not a guaranteed result of training.

## 5. Summarize new text

Use the trained adapter by default:

```bash
python scripts/summarize.py --text "A long article or document goes here."
python scripts/summarize.py --file article.txt
```

Compare with the untouched base model:

```bash
python scripts/summarize.py --base --file article.txt
```

## What LoRA and QLoRA mean

LoRA freezes the base model and trains small low-rank matrices in selected layers. The adapter is much smaller than a full model checkpoint and remains separate here. QLoRA applies LoRA training while the frozen base model is quantized; because this project starts from a 4-bit model, `mlx-lm` performs QLoRA-style training.

## Customize the experiment

- **Use real data:** replace `data/sample_raw.jsonl` with your own UTF-8 JSONL containing one non-empty `text` and `summary` pair per line, then rerun preparation. This is the main file to modify when replacing the synthetic examples.
- **Change the model:** update `model` in `configs/lora_config.yaml`, then pass the same identifier through `--model` to baseline, evaluation, and summarization. The code default is `DEFAULT_MODEL` in `scripts/common.py`; update it too if you want the new model to become every CLI's default.
- **Reduce memory use:** keep `batch_size: 1`, lower `max_seq_length` from 2048, or retain `grad_checkpoint: true` in the YAML. Shorter sequences usually have the largest memory impact.
- **Scale the data:** add hundreds or thousands of diverse, accurate pairs, preserve a genuinely held-out test split, and adjust iteration count so the larger dataset is not repeatedly overfit.
- **Tune generation:** all inference CLIs accept `--max-tokens` and `--temperature`. Use identical values for fair comparisons.

## Tests

```bash
python -m pytest -q
```

Unit tests cover data preparation, prompts, metrics, reports, wrapper commands, and CLI validation without downloading a model. Model download, generation, and training are separate smoke tests because they require Apple Silicon, several gigabytes of disk, and more time.

## Local test interface

Start the private, local-only comparison page:

```bash
python scripts/ui.py
```

Open `http://127.0.0.1:8000`, paste an English passage, and click **Compare summaries**. The page runs the untouched base model and `adapters/best` with identical settings, then displays both summaries and their generation times. Your text stays on this Mac. Stop the server with Control-C.

## Current mlx-lm interface

This MVP uses `mlx-lm[train]` 0.29.1 because it is the newest release that installs with the Apple-provided Python 3.9 and available MLX wheels on this test Mac. Newer `mlx-lm` releases require Python 3.10 or later. The installed 0.29.1 package supports the module command:

```bash
python -m mlx_lm lora --config configs/lora_config.yaml
```

The dataset uses MLX's chat format with the target summary as the final assistant message. `mask_prompt: true` makes the loss focus on that completion. No adapter fusion command is used.
