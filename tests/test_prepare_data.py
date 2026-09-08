import json
from pathlib import Path

import pytest

from scripts.common import SYSTEM_PROMPT, chat_record, extract_source_and_reference
from scripts.prepare_data import prepare


def write_raw(path: Path, count: int = 7) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            handle.write(json.dumps({"text": f"Document {index}", "summary": f"Summary {index}"}) + "\n")


def test_chat_record_uses_exact_instruction_contract():
    record = chat_record("A source.", "A summary.")
    assert record == {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Summarize the following text:\n\nA source."},
        {"role": "assistant", "content": "A summary."},
    ]}
    assert extract_source_and_reference(record) == ("A source.", "A summary.")


def test_prepare_is_deterministic_and_preserves_all_examples(tmp_path, capsys):
    raw = tmp_path / "raw.jsonl"
    write_raw(raw)
    first, second = tmp_path / "first", tmp_path / "second"

    counts = prepare(raw, first, seed=42, train_ratio=0.7, valid_ratio=0.15)
    prepare(raw, second, seed=42, train_ratio=0.7, valid_ratio=0.15)

    assert counts == {"train": 5, "valid": 1, "test": 1}
    assert "train: 5" in capsys.readouterr().out
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    rows = [json.loads(line) for name in ("train.jsonl", "valid.jsonl", "test.jsonl")
            for line in (first / name).read_text(encoding="utf-8").splitlines()]
    sources = [extract_source_and_reference(row)[0] for row in rows]
    assert len(sources) == len(set(sources)) == 7


@pytest.mark.parametrize("record", [
    {"text": "", "summary": "ok"},
    {"text": "ok", "summary": "   "},
    {"text": 123, "summary": "ok"},
    {"summary": "ok"},
])
def test_prepare_rejects_invalid_fields_before_writing(tmp_path, record):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="line 1"):
        prepare(raw, output, seed=1, train_ratio=0.8, valid_ratio=0.1)
    assert not output.exists()

