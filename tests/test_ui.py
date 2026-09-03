import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.ui import ComparisonService, create_server, validate_text


class FakeGenerator:
    def __init__(self, name):
        self.name = name


def test_validate_text_rejects_invalid_values():
    for value in (None, 3, "", "   "):
        with pytest.raises(ValueError):
            validate_text(value)
    with pytest.raises(ValueError, match="20,000"):
        validate_text("x" * 20_001)
    assert validate_text("  useful text  ") == "useful text"


def test_comparison_service_caches_models_and_uses_equal_settings():
    loads, calls = [], []

    def loader(model, adapter_path=None):
        loads.append((model, adapter_path))
        return FakeGenerator("lora" if adapter_path else "base")

    def generate(generator, text, **settings):
        calls.append((generator.name, text, settings))
        return f"{generator.name} summary"

    service = ComparisonService(loader=loader, generator=generate)
    first = service.compare("Document")
    second = service.compare("Another")

    assert loads == [(service.model_name, None), (service.model_name, service.adapter_path)]
    assert [item[2] for item in calls] == [
        {"max_tokens": 180, "temperature": 0.0},
        {"max_tokens": 180, "temperature": 0.0},
        {"max_tokens": 180, "temperature": 0.0},
        {"max_tokens": 180, "temperature": 0.0},
    ]
    assert first["base"]["summary"] == "base summary"
    assert first["lora"]["summary"] == "lora summary"
    assert first["base"]["seconds"] >= 0
    assert second["lora"]["seconds"] >= 0


@pytest.fixture
def running_server(tmp_path):
    for name, body in {
        "index.html": "<h1>LoRA Summarizer</h1>",
        "styles.css": "body { color: black; }",
        "app.js": "console.log('ready')",
    }.items():
        (tmp_path / name).write_text(body, encoding="utf-8")

    class StubService:
        def compare(self, text):
            clean = validate_text(text)
            return {"base": {"summary": clean, "seconds": 0.1},
                    "lora": {"summary": clean.upper(), "seconds": 0.2}}

    server = create_server("127.0.0.1", 0, StubService(), tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()
    server.server_close()


def test_http_server_serves_page_and_compare_api(running_server):
    assert "LoRA Summarizer" in urlopen(running_server + "/").read().decode()
    request = Request(running_server + "/api/compare",
                      data=json.dumps({"text": "hello"}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    payload = json.load(urlopen(request))
    assert payload["base"]["summary"] == "hello"
    assert payload["lora"]["summary"] == "HELLO"


def test_http_server_returns_json_400_for_blank_input(running_server):
    request = Request(running_server + "/api/compare",
                      data=b'{"text":" "}', headers={"Content-Type": "application/json"},
                      method="POST")
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 400
    assert "error" in json.load(error.value)
