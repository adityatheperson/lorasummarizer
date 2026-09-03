import sys
from pathlib import Path

from scripts.train import build_command, load_config


CONFIG = Path(__file__).parents[1] / "configs" / "lora_config.yaml"


def test_config_has_conservative_q_lora_settings():
    config = load_config(CONFIG)
    assert config["model"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert config["train"] is True
    assert config["fine_tune_type"] == "lora"
    assert config["data"] == "data"
    assert config["batch_size"] == 1
    assert config["iters"] == 300
    assert config["max_seq_length"] == 2048
    assert config["mask_prompt"] is True
    assert config["adapter_path"] == "adapters/summarizer"
    assert config["steps_per_eval"] == 25
    assert config["lora_parameters"]["rank"] == 8


def test_build_command_uses_current_module_cli_and_overrides():
    command = build_command(CONFIG, iters=5, adapter_path=Path("adapters/smoke"))
    assert command[:6] == [sys.executable, "-m", "mlx_lm", "lora", "--config", str(CONFIG)]
    assert command[6:] == ["--iters", "5", "--adapter-path", "adapters/smoke"]
