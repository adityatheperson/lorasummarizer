from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

try:
    from .common import DEFAULT_MODEL, generate_summary, load_generator
except ImportError:
    from common import DEFAULT_MODEL, generate_summary, load_generator


MAX_TEXT_LENGTH = 20_000
MAX_REQUEST_BYTES = 80_000


def validate_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Enter some English text to summarize.")
    clean = value.strip()
    if len(clean) > MAX_TEXT_LENGTH:
        raise ValueError("Text must be 20,000 characters or fewer.")
    return clean


class ComparisonService:
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 adapter_path: str | Path = Path("adapters/best"), *,
                 loader: Callable[..., Any] = load_generator,
                 generator: Callable[..., str] = generate_summary,
                 max_tokens: int = 180, temperature: float = 0.0):
        self.model_name = model_name
        self.adapter_path = Path(adapter_path)
        self.loader = loader
        self.generator = generator
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._base = None
        self._lora = None
        self._lock = threading.Lock()

    def _models(self) -> tuple[Any, Any]:
        if self._base is None:
            self._base = self.loader(self.model_name)
        if self._lora is None:
            self._lora = self.loader(self.model_name, self.adapter_path)
        return self._base, self._lora

    def compare(self, text: Any) -> dict[str, dict[str, str | float]]:
        clean = validate_text(text)
        with self._lock:
            base, lora = self._models()
            results = {}
            for key, model in (("base", base), ("lora", lora)):
                started = time.perf_counter()
                summary = self.generator(model, clean, max_tokens=self.max_tokens,
                                         temperature=self.temperature)
                results[key] = {"summary": summary,
                                "seconds": round(time.perf_counter() - started, 3)}
        return results


def create_server(host: str, port: int, service: ComparisonService,
                  ui_dir: str | Path) -> ThreadingHTTPServer:
    root = Path(ui_dir).resolve()
    allowed = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            asset = allowed.get(self.path)
            if asset is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                body = (root / asset[0]).read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", asset[1])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/api/compare":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("Request is empty or too large.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                self._json(HTTPStatus.OK, service.compare(payload.get("text")))
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                           {"error": "Summary generation failed. Check the terminal and try again."})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local LoRA comparison UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=Path("adapters/best"))
    parser.add_argument("--max-tokens", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ui_dir = Path(__file__).parents[1] / "ui"
    if not args.adapter_path.is_dir():
        raise SystemExit(f"error: adapter directory does not exist: {args.adapter_path}")
    service = ComparisonService(args.model, args.adapter_path, max_tokens=args.max_tokens)
    server = create_server(args.host, args.port, service, ui_dir)
    print(f"LoRA Summarizer is ready at http://{args.host}:{args.port}", flush=True)
    print("Press Control-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
