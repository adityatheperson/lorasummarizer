from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

try:
    from .common import DEFAULT_MODEL, extract_source_and_reference, generate_summary, load_generator, read_jsonl, write_jsonl
except ImportError:
    from common import DEFAULT_MODEL, extract_source_and_reference, generate_summary, load_generator, read_jsonl, write_jsonl


def run_dataset(test_path: str | Path, output_path: str | Path,
                summarizer: Callable[[str], str], limit: int | None = None) -> list[dict]:
    rows = []
    records = read_jsonl(test_path)
    if limit is not None:
        records = records[:limit]
    for index, record in enumerate(records, 1):
        source, reference = extract_source_and_reference(record)
        started = time.perf_counter()
        generated = summarizer(source)
        row = {"source": source, "reference": reference, "generated_summary": generated,
               "generation_seconds": time.perf_counter() - started}
        rows.append(row)
        print(f"Generated {index}/{len(records)}")
    write_jsonl(output_path, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate untouched base-model summaries.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--test-file", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        generator = load_generator(args.model)
        run_dataset(args.test_file, args.output,
                    lambda source: generate_summary(generator, source, max_tokens=args.max_tokens,
                                                    temperature=args.temperature), args.limit)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
