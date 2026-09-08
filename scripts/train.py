from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {source}")
    return config


def build_command(config_path: str | Path, *, iters: int | None = None,
                  adapter_path: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]
    if iters is not None:
        if iters < 1:
            raise ValueError("--iters must be at least 1")
        command.extend(["--iters", str(iters)])
    if adapter_path is not None:
        command.extend(["--adapter-path", str(adapter_path)])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mlx-lm QLoRA training.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_config.yaml"))
    parser.add_argument("--iters", type=int, help="Override iterations; use 5 for a smoke test")
    parser.add_argument("--adapter-path", type=Path, help="Override adapter output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
        chosen_iters = args.iters if args.iters is not None else config.get("iters")
        chosen_adapter = args.adapter_path or Path(config.get("adapter_path", "adapters"))
        print(f"Model: {config.get('model')}")
        print(f"Configuration: {args.config}")
        print(f"Iterations: {chosen_iters}")
        print(f"Adapter output: {chosen_adapter}")
        subprocess.run(build_command(args.config, iters=args.iters,
                                     adapter_path=args.adapter_path), check=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"error: training failed: {exc}") from exc


if __name__ == "__main__":
    main()
