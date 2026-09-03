import pytest

from scripts.metrics import rouge1_f1, rouge_l_f1, score, tokenize


def test_tokenize_is_case_insensitive_and_keeps_apostrophes():
    assert tokenize("Ada's MODEL, version 2!") == ["ada's", "model", "version", "2"]


def test_rouge1_counts_duplicate_tokens():
    assert rouge1_f1("red red blue", "red blue blue") == pytest.approx(2 / 3)


def test_rouge_l_uses_longest_common_subsequence():
    assert rouge_l_f1("a b c d", "a c d x") == pytest.approx(0.75)


@pytest.mark.parametrize("metric", [rouge1_f1, rouge_l_f1])
def test_metrics_handle_empty_and_perfect_text(metric):
    assert metric("", "anything") == 0.0
    assert metric("Same words", "same words") == 1.0


def test_score_returns_both_metrics():
    assert score("one two", "one") == {"rouge1": pytest.approx(2 / 3), "rougeL": pytest.approx(2 / 3)}
