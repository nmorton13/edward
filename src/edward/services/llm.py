"""Decoupled OpenAI-compatible model client for synthesis, answers, and finding extraction.

Supports separate Answer and Embedding configurations, strict Pydantic output
validation with bounded correction retries, privacy policy enforcement, and diagnostic logging.
"""

import json
import logging
import os
import re
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from edward.services.diagnostics import record_model_diagnostic
from edward.services.privacy import (
    DataClass,
    assert_transmission_permitted,
    resolve_provider_location,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base error for LLM client failures."""

    pass


class LLMConnectionError(LLMError):
    """Raised when unable to connect to LLM endpoint."""

    pass


class LLMResponseValidationError(LLMError):
    """Raised when LLM response cannot be validated against the requested Pydantic schema."""

    def __init__(self, message: str, raw_output: str, diagnostic_path: str | None = None):
        super().__init__(message)
        self.raw_output = raw_output
        self.diagnostic_path = diagnostic_path


def extract_json_from_text(text: str) -> str:
    """Extract a JSON object or array from model response text, stripping markdown fences if present."""
    text = text.strip()
    # Check for markdown code blocks ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try finding first { or [ to last } or ]
    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        last_brace = text.rfind("}")
        if last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1].strip()
    elif first_bracket != -1:
        last_bracket = text.rfind("]")
        if last_bracket != -1 and last_bracket > first_bracket:
            return text[first_bracket : last_bracket + 1].strip()

    return text


class LLMClient:
    """OpenAI-compatible chat completion client."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
        timeout: float = 30.0,
        provider: str = "llm",
    ):
        raw_base = (
            base_url or os.environ.get("EDWARD_ANSWER_BASE_URL") or "http://localhost:11434/v1"
        ).rstrip("/")
        self.base_url = raw_base
        self.model = model or os.environ.get("EDWARD_ANSWER_MODEL") or "llama3.2"
        self.api_key = api_key or os.environ.get("EDWARD_ANSWER_API_KEY", "")
        self.provider = provider or os.environ.get("EDWARD_ANSWER_PROVIDER", "llm")
        self.location = resolve_provider_location(
            provider_name=self.provider,
            declared_location=location or os.environ.get("EDWARD_ANSWER_LOCATION"),
            base_url=self.base_url,
        )
        self.timeout = timeout

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        response_model: type[T] | None = None,
        data_class: DataClass | list[DataClass] | set[DataClass] = "public_web",
        temperature: float = 0.2,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ) -> tuple[str, T | None]:
        """Send chat completion request, enforce privacy, and validate response schema.

        Returns (raw_content, parsed_model_or_none).
        """
        # 1. Privacy check MUST run and halt before request serialization or network dispatch
        classes = [data_class] if isinstance(data_class, str) else list(data_class)
        for dc in classes:
            assert_transmission_permitted(
                provider_name=self.provider,
                data_class=dc,
                declared_location=self.location,
                base_url=self.base_url,
            )

        url = f"{self.base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        current_messages = list(messages)
        last_raw_text = ""
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            payload = {
                "model": self.model,
                "messages": current_messages,
                "temperature": temperature,
            }

            try:
                if client is not None:
                    resp = client.post(url, json=payload, headers=headers, timeout=self.timeout)
                else:
                    with httpx.Client(timeout=self.timeout) as http_client:
                        resp = http_client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as e:
                raise LLMConnectionError(f"LLM request timed out: {e}") from e
            except httpx.RequestError as e:
                raise LLMConnectionError(f"LLM connection error: {e}") from e

            if resp.status_code != 200:
                raise LLMConnectionError(
                    f"LLM endpoint returned status {resp.status_code}: {resp.text[:200]}"
                )

            try:
                data = resp.json()
                last_raw_text = data["choices"][0]["message"]["content"]
            except Exception as e:
                raise LLMError(f"Failed to parse completion JSON response: {e}") from e

            if response_model is None:
                return last_raw_text, None

            # Attempt schema validation
            try:
                json_str = extract_json_from_text(last_raw_text)
                parsed_dict = json.loads(json_str)
                parsed_model = response_model.model_validate(parsed_dict)
                return last_raw_text, parsed_model
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.debug(
                    "Schema validation failed on attempt %d: %s. Raw text: %s",
                    attempt + 1,
                    e,
                    last_raw_text[:200],
                )
                if attempt < max_retries:
                    # Append correction prompt
                    current_messages.append({"role": "assistant", "content": last_raw_text})
                    current_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous response had a schema/formatting error: {e}. "
                                f"Please output strictly valid JSON conforming to the requested schema, "
                                f"with no surrounding commentary or explanation."
                            ),
                        }
                    )

        # All retries failed: record diagnostic and raise
        diag_dc = "public_web"
        if isinstance(data_class, str):
            diag_dc = data_class
        else:
            for priority_dc in ("gmail", "personal_notes", "documents", "public_web"):
                if priority_dc in data_class:
                    diag_dc = priority_dc
                    break
        diag_path = record_model_diagnostic(
            provider=f"llm-{self.model}",
            raw_output=last_raw_text,
            error_message=str(last_error),
            data_class=diag_dc,
            context={
                "model": self.model,
                "schema": response_model.__name__ if response_model else None,
            },
        )
        raise LLMResponseValidationError(
            f"Model output failed validation against {response_model.__name__ if response_model else 'schema'}: {last_error}",
            raw_output=last_raw_text,
            diagnostic_path=str(diag_path) if diag_path else None,
        )


def get_answer_client() -> LLMClient | None:
    """Return a configured LLMClient for answering, or None.

    Edward is memory, not a mind: no model is contacted unless an operator
    explicitly opts in. Absence is the default, so a leftover base URL or model
    name from a previous configuration cannot silently re-enable a model.
    """
    mode = os.environ.get("EDWARD_ANSWERER_MODE", "disabled").strip().lower() or "disabled"
    if mode in ("disabled", "none", "off", ""):
        return None

    local_modes = ("local-model", "local")
    enable_only = mode in ("enabled", "on")

    # Check if base URL or model is specified
    base_url = os.environ.get("EDWARD_ANSWER_BASE_URL")
    model = os.environ.get("EDWARD_ANSWER_MODEL")

    # A local mode may fall back to the localhost default; anything else needs an
    # explicit endpoint or model to be usable.
    if not base_url and not model and mode not in local_modes and not enable_only:
        return None

    timeout = float(os.environ.get("EDWARD_ANSWER_TIMEOUT", "120"))
    if timeout <= 0:
        raise ValueError("EDWARD_ANSWER_TIMEOUT must be greater than zero")

    return LLMClient(
        base_url=base_url,
        model=model,
        location=os.environ.get("EDWARD_ANSWER_LOCATION"),
        timeout=timeout,
    )
