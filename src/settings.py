"""
Shared runtime settings.

APP_LANGUAGE controls user-facing language:
- cn: Chinese output, default
- en: English output
"""

from __future__ import annotations

import os


def get_app_language() -> str:
    """
    Return the configured app language.

    Only cn and en are supported. Invalid or missing values fall back to cn.
    """
    language = os.getenv("APP_LANGUAGE", "cn").strip().lower()

    if language in {"cn", "en"}:
        return language

    return "cn"


def is_english() -> bool:
    return get_app_language() == "en"


def output_language_instruction() -> str:
    """
    Prompt instruction used by LLM agents.
    """
    if is_english():
        return "Output language: English. All narrative text fields must be written in English."

    return "输出语言：中文。所有分析性文本字段必须使用中文。"
