from scripts.evaluate import render_comparison


def test_render_comparison_contains_examples_and_averages():
    rows = [{
        "source": "A source",
        "reference": "A reference",
        "base_summary": "Base output",
        "lora_summary": "LoRA output",
        "base_metrics": {"rouge1": 0.2, "rougeL": 0.1},
        "lora_metrics": {"rouge1": 0.5, "rougeL": 0.4},
    }]
    report = render_comparison(rows, {"rouge1": 0.2, "rougeL": 0.1},
                               {"rouge1": 0.5, "rougeL": 0.4})
    for text in ("SOURCE:", "REFERENCE:", "BASE MODEL:", "LORA MODEL:", "A source", "LoRA output"):
        assert text in report
    assert "| Base model | 0.2000 | 0.1000 |" in report
    assert "| LoRA model | 0.5000 | 0.4000 |" in report
