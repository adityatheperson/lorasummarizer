from __future__ import annotations

import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def _f1(overlap: int, reference_count: int, prediction_count: int) -> float:
    if not overlap or not reference_count or not prediction_count:
        return 0.0
    precision = overlap / prediction_count
    recall = overlap / reference_count
    return 2 * precision * recall / (precision + recall)


def rouge1_f1(reference: str, prediction: str) -> float:
    reference_tokens, prediction_tokens = tokenize(reference), tokenize(prediction)
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    return _f1(overlap, len(reference_tokens), len(prediction_tokens))


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, 1):
            current.append(previous[index - 1] + 1 if left_token == right_token
                           else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    reference_tokens, prediction_tokens = tokenize(reference), tokenize(prediction)
    return _f1(_lcs_length(reference_tokens, prediction_tokens),
               len(reference_tokens), len(prediction_tokens))


def score(reference: str, prediction: str) -> dict[str, float]:
    return {"rouge1": rouge1_f1(reference, prediction),
            "rougeL": rouge_l_f1(reference, prediction)}
