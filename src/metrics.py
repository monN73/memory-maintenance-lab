"""Deterministic text normalization, lexical overlap metrics, and token estimation."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any

NORMALIZATION_VERSION = "nfkc-char-bigram-v1"
TOKENIZER_VERSION = "approx-utf8-v1"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Unicode NFKC normalization with all whitespace stripped; preserves case, digits, punctuation."""
    nfkc = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in nfkc if not ch.isspace())


def char_bigram_list(text: str) -> list[str]:
    norm = normalize_text(text)
    if not norm:
        return []
    if len(norm) == 1:
        return [norm]
    return [norm[i : i + 2] for i in range(len(norm) - 1)]


def char_bigram_set(text: str) -> set[str]:
    return set(char_bigram_list(text))


def lexical_jaccard(before: str, after: str) -> tuple[float | None, str | None]:
    sa = char_bigram_set(before)
    sb = char_bigram_set(after)
    union = sa | sb
    if not union:
        return None, "empty_union"
    return round(len(sa & sb) / len(union), 6), None


def lexical_retention(before: str, after: str) -> tuple[float | None, str | None]:
    sa = char_bigram_set(before)
    if not sa:
        return None, "empty_before"
    sb = char_bigram_set(after)
    return round(len(sa & sb) / len(sa), 6), None


def duplicate_ngram_rate(text: str) -> tuple[float | None, str | None]:
    grams = char_bigram_list(text)
    if not grams:
        return None, "empty_text"
    return round(1.0 - (len(set(grams)) / len(grams)), 6), None


def estimate_text_tokens(text: str) -> int:
    """Deterministic token estimator (~1.5 chars/token for CJK, ~3.8 chars/token for ASCII)."""
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii = len(text) - ascii_chars
    return max(1, math.ceil(ascii_chars / 3.8 + non_ascii / 1.5))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Include per-message framing overhead (+4 tokens per message + 3 reply priming)."""
    total = 3
    for m in messages:
        total += 4 + estimate_text_tokens(m.get("role", "")) + estimate_text_tokens(m.get("content", ""))
    return total
