from __future__ import annotations

from typing import Any

from agentdojo.logging import Logger


def _usage_value(usage: Any, key: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(key) or 0)
    return int(getattr(usage, key, 0) or 0)


def _first_usage_value(usage: Any, *keys: str) -> int:
    for key in keys:
        value = _usage_value(usage, key)
        if value:
            return value
    return 0


def record_token_usage_from_response(response: Any, label: str = "unknown") -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    prompt_tokens = _usage_value(usage, "prompt_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(usage, "total_tokens") or prompt_tokens + completion_tokens
    record_token_usage(prompt_tokens, completion_tokens, total_tokens, label)


def record_token_usage_from_google_response(response: Any, label: str = "unknown") -> None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        usage = getattr(response, "usage", None)
    if usage is None:
        return

    prompt_tokens = _first_usage_value(usage, "prompt_token_count", "input_token_count", "prompt_tokens")
    completion_tokens = _first_usage_value(
        usage,
        "candidates_token_count",
        "candidate_token_count",
        "output_token_count",
        "completion_tokens",
    )
    total_tokens = _first_usage_value(usage, "total_token_count", "total_tokens") or prompt_tokens + completion_tokens
    record_token_usage(prompt_tokens, completion_tokens, total_tokens, label)


def record_token_usage_from_anthropic_response(response: Any, label: str = "unknown") -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    prompt_tokens = _first_usage_value(usage, "input_tokens", "prompt_tokens")
    completion_tokens = _first_usage_value(usage, "output_tokens", "completion_tokens")
    total_tokens = _first_usage_value(usage, "total_tokens") or prompt_tokens + completion_tokens
    record_token_usage(prompt_tokens, completion_tokens, total_tokens, label)


def record_token_usage_from_bedrock_response(response: Any, label: str = "unknown") -> None:
    usage = response.get("usage", {}) if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return

    prompt_tokens = _first_usage_value(usage, "inputTokens", "input_tokens", "prompt_tokens")
    completion_tokens = _first_usage_value(usage, "outputTokens", "output_tokens", "completion_tokens")
    total_tokens = _first_usage_value(usage, "totalTokens", "total_tokens") or prompt_tokens + completion_tokens
    record_token_usage(prompt_tokens, completion_tokens, total_tokens, label)


def record_token_usage_from_cohere_response(response: Any, label: str = "unknown") -> None:
    meta = getattr(response, "meta", None)
    if meta is None:
        return

    tokens = getattr(meta, "tokens", None)
    if tokens is None and isinstance(meta, dict):
        tokens = meta.get("tokens")
    billed_units = getattr(meta, "billed_units", None)
    if billed_units is None and isinstance(meta, dict):
        billed_units = meta.get("billed_units")

    usage = tokens or billed_units
    if usage is None:
        return

    prompt_tokens = _first_usage_value(usage, "input_tokens", "prompt_tokens")
    completion_tokens = _first_usage_value(usage, "output_tokens", "completion_tokens")
    total_tokens = _first_usage_value(usage, "total_tokens") or prompt_tokens + completion_tokens
    record_token_usage(prompt_tokens, completion_tokens, total_tokens, label)


def record_token_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int, label: str = "unknown") -> None:
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return

    logger = Logger.get()
    if not hasattr(logger, "context") or not hasattr(logger, "set_contextarg"):
        return

    current = dict(getattr(logger, "context", {}).get("token_usage") or {})
    by_label = dict(current.get("by_label") or {})
    label_usage = dict(by_label.get(label) or {})

    for target in (current, label_usage):
        target["prompt_tokens"] = int(target.get("prompt_tokens") or 0) + prompt_tokens
        target["completion_tokens"] = int(target.get("completion_tokens") or 0) + completion_tokens
        target["total_tokens"] = int(target.get("total_tokens") or 0) + total_tokens
        target["calls"] = int(target.get("calls") or 0) + 1

    by_label[label] = label_usage
    current["by_label"] = by_label
    logger.set_contextarg("token_usage", current)
