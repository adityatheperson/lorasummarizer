from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .baseline import run_dataset
    from .common import DEFAULT_MODEL, generate_summary, load_generator, read_jsonl, write_jsonl
    from .metrics import score
except ImportError:
    from baseline import run_dataset
    from common import DEFAULT_MODEL, generate_summary, load_generator, read_jsonl, write_jsonl
    from metrics import score


def render_comparison(rows: list[dict[str, Any]], base_average: dict[str, float],
                      lora_average: dict[str, float]) -> str:
    sections = ["# Base vs. LoRA Summarization Comparison", ""]
    for index, row in enumerate(rows, 1):
        sections.extend([
            f"## Example {index}", "", "SOURCE:", "", row["source"], "",
            "REFERENCE:", "", row["reference"], "", "BASE MODEL:", "",
            row["base_summary"], "", "LORA MODEL:", "", row["lora_summary"], "",
            "| Output | ROUGE-1 F1 | ROUGE-L F1 |",
            "|---|---:|---:|",
            f"| Base | {row['base_metrics']['rouge1']:.4f} | {row['base_metrics']['rougeL']:.4f} |",
            f"| LoRA | {row['lora_metrics']['rouge1']:.4f} | {row['lora_metrics']['rougeL']:.4f} |", "",
        ])
    sections.extend([
        "## Average metrics", "", "| Model | ROUGE-1 F1 | ROUGE-L F1 |",
        "|---|---:|---:|",
        f"| Base model | {base_average['rouge1']:.4f} | {base_average['rougeL']:.4f} |",
        f"| LoRA model | {lora_average['rouge1']:.4f} | {lora_average['rougeL']:.4f} |", "",
    ])
    return "\n".join(sections)


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"rouge1": 0.0, "rougeL": 0.0}
    return {key: sum(row[key] for row in rows) / len(rows) for key in ("rouge1", "rougeL")}


def evaluate(test_file: Path, baseline_file: Path, lora_file: Path, report_file: Path,
             model_name: str, adapter_path: Path, max_tokens: int, temperature: float,
             reuse_baseline: bool = True) -> tuple[dict[str, float], dict[str, float]]:
    if reuse_baseline and baseline_file.is_file():
        base_rows = read_jsonl(baseline_file)
    else:
        base_generator = load_generator(model_name)
        base_rows = run_dataset(test_file, baseline_file,
                                lambda source: generate_summary(base_generator, source,
                                                                max_tokens=max_tokens,
                                                                temperature=temperature))
    lora_generator = load_generator(model_name, adapter_path)
    lora_rows = run_dataset(test_file, lora_file,
                            lambda source: generate_summary(lora_generator, source,
                                                            max_tokens=max_tokens,
                                                            temperature=temperature))
    if len(base_rows) != len(lora_rows):
        raise ValueError("Baseline and LoRA result counts differ; regenerate the baseline")
    comparisons, base_scores, lora_scores = [], [], []
    for base, lora in zip(base_rows, lora_rows):
        if base["source"] != lora["source"] or base["reference"] != lora["reference"]:
            raise ValueError("Baseline results do not match the current test set")
        base_metric = score(base["reference"], base["generated_summary"])
        lora_metric = score(lora["reference"], lora["generated_summary"])
        base_scores.append(base_metric)
        lora_scores.append(lora_metric)
        comparisons.append({"source": base["source"], "reference": base["reference"],
                            "base_summary": base["generated_summary"],
                            "lora_summary": lora["generated_summary"],
                            "base_metrics": base_metric, "lora_metrics": lora_metric})
    base_average, lora_average = _average(base_scores), _average(lora_scores)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_comparison(comparisons, base_average, lora_average), encoding="utf-8")
    return base_average, lora_average


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare base and LoRA summarization quality.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=Path("adapters/summarizer"))
    parser.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--baseline-output", type=Path, default=Path("outputs/baseline.jsonl"))
    parser.add_argument("--lora-output", type=Path, default=Path("outputs/lora.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("outputs/comparison.md"))
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--regenerate-baseline", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if not args.adapter_path.is_dir():
            raise ValueError(f"Adapter directory does not exist: {args.adapter_path}")
        base, lora = evaluate(args.test_file, args.baseline_output, args.lora_output,
                              args.report, args.model, args.adapter_path, args.max_tokens,
                              args.temperature, not args.regenerate_baseline)
        print(f"Base model: ROUGE-1={base['rouge1']:.4f}, ROUGE-L={base['rougeL']:.4f}")
        print(f"LoRA model: ROUGE-1={lora['rouge1']:.4f}, ROUGE-L={lora['rougeL']:.4f}")
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
