import os
from typing import Any


DEFAULT_OPENAI_TIMEOUT_SECONDS = 180.0
DEFAULT_OPENAI_MAX_RETRIES = 8
DEFAULT_OPENAI_TENACITY_ATTEMPTS = 4


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def openai_timeout_seconds() -> float:
    return _env_float("AGENTDOJO_OPENAI_TIMEOUT", DEFAULT_OPENAI_TIMEOUT_SECONDS)


def openai_max_retries() -> int:
    return _env_int("AGENTDOJO_OPENAI_MAX_RETRIES", DEFAULT_OPENAI_MAX_RETRIES)


def openai_tenacity_attempts() -> int:
    return _env_int("AGENTDOJO_OPENAI_TENACITY_ATTEMPTS", DEFAULT_OPENAI_TENACITY_ATTEMPTS)


def openai_client_kwargs(**kwargs: Any) -> dict[str, Any]:
    return {
        **kwargs,
        "timeout": kwargs.get("timeout", openai_timeout_seconds()),
        "max_retries": kwargs.get("max_retries", openai_max_retries()),
    }
