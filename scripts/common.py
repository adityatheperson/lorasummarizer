from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
SYSTEM_PROMPT = (
    "You are a concise English text summarizer. Preserve the most important facts, "
    "names, numbers, conclusions, and relationships. Do not introduce information "
    "that is not present in the source."
)


def user_prompt(text: str) -> str:
    return f"Summarize the following text:\n\n{text}"


def chat_record(text: str, summary: str) -> dict[str, list[dict[str, str]]]:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(text)},
        {"role": "assistant", "content": summary},
    ]}


def extract_source_and_reference(record: dict[str, Any]) -> tuple[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Expected a three-message MLX chat record")
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected system, user, assistant role order")
    prefix = "Summarize the following text:\n\n"
    content = messages[1].get("content", "")
    if not isinstance(content, str) or not content.startswith(prefix):
        raise ValueError("User message does not contain the summarization prompt")
    reference = messages[2].get("content", "")
    if not isinstance(reference, str):
        raise ValueError("Assistant summary must be text")
    return content[len(prefix):], reference


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {source}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {source}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class Generator:
    model: Any
    tokenizer: Any
    generate: Callable[..., str]
    make_sampler: Callable[..., Any]


def load_generator(model_name: str = DEFAULT_MODEL, adapter_path: str | Path | None = None) -> Generator:
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError("mlx-lm is not installed; run pip install -r requirements.txt") from exc
    kwargs = {"adapter_path": str(adapter_path)} if adapter_path is not None else {}
    model, tokenizer = load(model_name, **kwargs)
    return Generator(model, tokenizer, generate, make_sampler)


def generate_summary(generator: Generator, source: str, *, max_tokens: int = 180,
                     temperature: float = 0.0) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(source)},
    ]
    prompt = generator.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    sampler = generator.make_sampler(temp=temperature)
    output = generator.generate(generator.model, generator.tokenizer, prompt=prompt,
                                max_tokens=max_tokens, sampler=sampler, verbose=False)
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL | re.IGNORECASE)
    output = re.sub(r"</?tool_call>", "", output, flags=re.IGNORECASE)
    return output.strip()
