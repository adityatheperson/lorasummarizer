from __future__ import annotations

import argparse
import random
from pathlib import Path

try:
    from .common import chat_record, read_jsonl, write_jsonl
except ImportError:
    from common import chat_record, read_jsonl, write_jsonl


def _validated_pairs(input_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line_number, row in enumerate(read_jsonl(input_path), 1):
        text, summary = row.get("text"), row.get("summary")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid or empty 'text' on line {line_number} of {input_path}")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(f"Invalid or empty 'summary' on line {line_number} of {input_path}")
        pairs.append((text.strip(), summary.strip()))
    if len(pairs) < 3:
        raise ValueError("At least 3 examples are required to create train, valid, and test splits")
    return pairs


def prepare(input_path: str | Path, output_dir: str | Path, *, seed: int = 42,
            train_ratio: float = 0.8, valid_ratio: float = 0.1) -> dict[str, int]:
    if not 0 < train_ratio < 1 or not 0 < valid_ratio < 1 or train_ratio + valid_ratio >= 1:
        raise ValueError("Ratios must be positive and train_ratio + valid_ratio must be less than 1")
    pairs = _validated_pairs(Path(input_path))
    random.Random(seed).shuffle(pairs)
    total = len(pairs)
    valid_count = max(1, round(total * valid_ratio))
    test_count = max(1, total - round(total * (train_ratio + valid_ratio)))
    train_count = total - valid_count - test_count
    if train_count < 1:
        raise ValueError("Ratios leave no training examples")
    splits = {
        "train": pairs[:train_count],
        "valid": pairs[train_count:train_count + valid_count],
        "test": pairs[train_count + valid_count:],
    }
    destination = Path(output_dir)
    for name, split in splits.items():
        write_jsonl(destination / f"{name}.jsonl", (chat_record(text, summary) for text, summary in split))
        print(f"{name}: {len(split)}")
    return {name: len(split) for name, split in splits.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare deterministic MLX chat-format data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        prepare(args.input, args.output_dir, seed=args.seed,
                train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
