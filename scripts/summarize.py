from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import DEFAULT_MODEL, generate_summary, load_generator
except ImportError:
    from common import DEFAULT_MODEL, generate_summary, load_generator


def read_source(text: str | None, file_path: Path | None) -> str:
    if (text is None) == (file_path is None):
        raise ValueError("Provide exactly one of --text or --file")
    if file_path is not None:
        if not file_path.is_file():
            raise ValueError(f"Input file does not exist: {file_path}")
        text = file_path.read_text(encoding="utf-8")
    assert text is not None
    if not text.strip():
        raise ValueError("Input text is empty")
    return text.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize English text with the LoRA adapter.")
    parser.add_argument("--text")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=Path("adapters/summarizer"))
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        source = read_source(args.text, args.file)
        adapter = None if args.base else args.adapter_path
        if adapter is not None and not adapter.is_dir():
            raise ValueError(f"Adapter directory does not exist: {adapter}. Train first or use --base.")
        generator = load_generator(args.model, adapter)
        print(generate_summary(generator, source, max_tokens=args.max_tokens,
                               temperature=args.temperature))
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
