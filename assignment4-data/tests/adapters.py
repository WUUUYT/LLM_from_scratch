from __future__ import annotations

import os
from typing import Any

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filter import passes_gopher_quality_filter
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language
from cs336_data.mask_pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality_classifier import classify_quality

LANG_REMAP = {
    "zho": "zh",
    "cmn": "zh",
    # 其他需要映射的语言可以在这里添加
}


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text_from_html_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    lang, score = identify_language(text)
    # 应用映射
    lang = LANG_REMAP.get(lang, lang)
    return (lang, score)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_numbers(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ips(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return classify_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return classify_toxic_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    return passes_gopher_quality_filter(text)


def run_exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
