"""OpenAI SDK error classification helpers."""

from __future__ import annotations

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)

RETRYABLE_OPENAI_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)
NON_RETRYABLE_OPENAI_ERRORS = (BadRequestError, AuthenticationError, PermissionDeniedError)
GENERAL_OPENAI_ERRORS = (APIStatusError, APIError)


def compact_error(error: Exception, limit: int = 500) -> str:
    text = str(error).strip() or error.__class__.__name__
    return text[:limit] + "..." if len(text) > limit else text


def error_summary(category: str, error: Exception) -> str:
    return f"{category}: {compact_error(error)}"
