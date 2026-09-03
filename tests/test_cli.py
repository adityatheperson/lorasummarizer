import json
from pathlib import Path

import pytest

from scripts.baseline import run_dataset
from scripts.common import Generator, chat_record, generate_summary
from scripts.summarize import read_source


class FakeTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"add_generation_prompt": True, "tokenize": False}
        return "FORMATTED PROMPT"


def test_generate_summary_uses_shared_chat_prompt():
    tokenizer = FakeTokenizer()
    calls = []
    generator = Generator(object(), tokenizer,
                          lambda model, tokenizer, prompt, **kwargs: calls.append((prompt, kwargs)) or " result ",
                          lambda **kwargs: ("sampler", kwargs))
    assert generate_summary(generator, "Source", max_tokens=12, temperature=0.0) == "result"
    assert tokenizer.messages[0]["role"] == "system"
    assert tokenizer.messages[1]["content"] == "Summarize the following text:\n\nSource"
    assert calls == [("FORMATTED PROMPT", {"max_tokens": 12,
                                             "sampler": ("sampler", {"temp": 0.0}),
                                             "verbose": False})]


def test_run_dataset_writes_required_fields(tmp_path):
    test_file = tmp_path / "test.jsonl"
    test_file.write_text(json.dumps(chat_record("Source", "Reference")) + "\n", encoding="utf-8")
    output = tmp_path / "baseline.jsonl"
    run_dataset(test_file, output, lambda source: "Generated")
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["source"] == "Source"
    assert row["reference"] == "Reference"
    assert row["generated_summary"] == "Generated"
    assert isinstance(row["generation_seconds"], float)


def test_read_source_requires_exactly_one_input(tmp_path):
    article = tmp_path / "article.txt"
    article.write_text("From file", encoding="utf-8")
    assert read_source("Inline", None) == "Inline"
    assert read_source(None, article) == "From file"
    with pytest.raises(ValueError, match="exactly one"):
        read_source(None, None)
    with pytest.raises(ValueError, match="exactly one"):
        read_source("Inline", article)
    with pytest.raises(ValueError, match="empty"):
        read_source("  ", None)


def test_read_source_reports_missing_file(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        read_source(None, tmp_path / "missing.txt")
