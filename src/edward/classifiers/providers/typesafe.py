"""Direct TypeSafe System One provider transport.

Forced to location = 'hosted'. Dispatches requests to TypeSafe API with strict
privacy policy enforcement, defense against malformed payloads, and robust error handling.
"""

import logging
import os
from typing import Any

import httpx

from edward.services.privacy import DataClass, assert_transmission_permitted

logger = logging.getLogger(__name__)


class TypeSafeProviderError(Exception):
    """Base error for TypeSafe provider failures."""

    pass


class TypeSafeAuthenticationError(TypeSafeProviderError):
    """Raised on 401 Unauthorized."""

    pass


class TypeSafeRateLimitError(TypeSafeProviderError):
    """Raised on 429 Rate Limit with optional retry_after."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TypeSafeResponseValidationError(TypeSafeProviderError):
    """Raised when TypeSafe response violates schema, bounds, or probability sum constraints."""

    pass


class TypeSafeProvider:
    """TypeSafe System One API client."""

    # Invariant: TypeSafe is ALWAYS hosted
    location: str = "hosted"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 15.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "").strip()
        self.base_url = (
            base_url or os.environ.get("EDWARD_CLASSIFIER_BASE_URL", "https://api.typesafe.ai")
        ).rstrip("/")
        self.model = model or os.environ.get("EDWARD_CLASSIFIER_MODEL", "typesafe/jev-latest")
        self.timeout = timeout

    def evaluate_questions(
        self,
        text: str,
        questions: list[dict[str, Any]],
        data_class: DataClass = "public_web",
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        """Evaluate System One questions against text.

        Enforces privacy assertions BEFORE request payload serialization or network dispatch.
        """
        # 1. Privacy Enforcement (must run and halt BEFORE serialization/dispatch)
        assert_transmission_permitted(
            provider_name="typesafe",
            data_class=data_class,
            declared_location=self.location,
            base_url=self.base_url,
        )

        if not self.api_key:
            raise TypeSafeAuthenticationError(
                "Missing TYPESAFE_API_KEY environment variable for TypeSafe provider."
            )

        url = f"{self.base_url}/v1/systemone"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "edward-classifier/0.1.0",
        }
        payload = {
            "model": self.model,
            "text": text,
            "questions": questions,
        }

        # 2. HTTP Network Dispatch
        try:
            if client is not None:
                resp = client.post(url, json=payload, headers=headers, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as http_client:
                    resp = http_client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as e:
            raise TypeSafeProviderError(f"TypeSafe request timed out: {e}") from e
        except httpx.RequestError as e:
            raise TypeSafeProviderError(f"TypeSafe connection error: {e}") from e

        # 3. HTTP Status Code Handling
        if resp.status_code == 401:
            raise TypeSafeAuthenticationError("TypeSafe authentication failed (401 Unauthorized).")
        elif resp.status_code == 429:
            retry_after_str = resp.headers.get("Retry-After")
            retry_after = (
                float(retry_after_str) if retry_after_str and retry_after_str.isdigit() else None
            )
            raise TypeSafeRateLimitError(
                f"TypeSafe rate limit exceeded (429). Retry after: {retry_after}s",
                retry_after=retry_after,
            )
        elif resp.status_code == 408:
            raise TypeSafeProviderError("TypeSafe request timeout (408).")
        elif resp.status_code >= 500:
            raise TypeSafeProviderError(
                f"TypeSafe server error ({resp.status_code}): {resp.text[:200]}"
            )
        elif resp.status_code != 200:
            raise TypeSafeProviderError(
                f"TypeSafe returned unexpected status {resp.status_code}: {resp.text[:200]}"
            )

        # 4. JSON Response Parsing & Validation
        try:
            data = resp.json()
        except Exception as e:
            raise TypeSafeResponseValidationError(
                f"TypeSafe response was not valid JSON: {e}"
            ) from e

        if not isinstance(data, dict):
            raise TypeSafeResponseValidationError("TypeSafe response must be a JSON object.")

        answers_raw = data.get("answers")
        if answers_raw is None:
            # Check if root is answers dict
            if any(k in data for k in ("choice", "noul", "score")):
                answers_raw = [data]
            else:
                raise TypeSafeResponseValidationError("Missing 'answers' in TypeSafe response.")

        # If answers is a list of answer objects, convert to dict keyed by question_id
        validated_answers: dict[str, dict[str, Any]] = {}
        if isinstance(answers_raw, list):
            for ans in answers_raw:
                qid = ans.get("question_id") or ans.get("id")
                if not qid:
                    raise TypeSafeResponseValidationError(
                        "Answer item missing 'question_id' or 'id'."
                    )
                self._validate_answer_item(ans)
                validated_answers[qid] = ans
        elif isinstance(answers_raw, dict):
            for qid, ans in answers_raw.items():
                if not isinstance(ans, dict):
                    raise TypeSafeResponseValidationError(f"Answer for {qid} must be a dictionary.")
                self._validate_answer_item(ans)
                validated_answers[qid] = ans
        else:
            raise TypeSafeResponseValidationError("'answers' must be a list or dictionary.")

        # Extract usage and metadata
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        cost = float(usage.get("cost") or 0.0)

        resolved_model = str(data.get("model") or self.model)
        request_id = str(data.get("id") or resp.headers.get("x-request-id") or "")

        return {
            "provider": "typesafe",
            "requested_model": self.model,
            "resolved_model": resolved_model,
            "provider_request_id": request_id,
            "answers": validated_answers,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }

    def _validate_answer_item(self, ans: dict[str, Any]) -> None:
        """Validate an individual answer item for primitive conformance and probability bounds."""
        primitive = ans.get("primitive")
        if not primitive or primitive not in ("choice", "noul", "score"):
            raise TypeSafeResponseValidationError(
                f"Invalid primitive in answer: '{primitive}'. Expected 'choice', 'noul', or 'score'."
            )

        if primitive == "choice":
            choice = ans.get("choice")
            if not choice or not isinstance(choice, str):
                raise TypeSafeResponseValidationError(
                    "Choice answer must have non-empty string 'choice'."
                )
            probs = ans.get("probabilities")
            if probs is not None:
                if not isinstance(probs, dict):
                    raise TypeSafeResponseValidationError("Choice 'probabilities' must be a dict.")
                for opt, p in probs.items():
                    if not isinstance(p, (int, float)) or p < 0.0 or p > 1.0:
                        raise TypeSafeResponseValidationError(
                            f"Probability for option '{opt}' ({p}) outside [0.0, 1.0]."
                        )
                # Check sum close to 1.0
                total_p = sum(probs.values())
                if abs(total_p - 1.0) > 0.15:
                    raise TypeSafeResponseValidationError(
                        f"Choice probabilities must sum to approximately 1.0; got {total_p:.3f}."
                    )

        elif primitive == "noul":
            val = ans.get("noul")
            if val is None:
                val = ans.get("probability")
            if val is None or not isinstance(val, (int, float)) or val < 0.0 or val > 1.0:
                raise TypeSafeResponseValidationError(
                    f"Noul probability value ({val}) must be a float in [0.0, 1.0]."
                )

        elif primitive == "score":
            score = ans.get("score")
            max_score = ans.get("max_score", 1.0)
            if score is None or not isinstance(score, (int, float)):
                raise TypeSafeResponseValidationError("Score answer must have numeric 'score'.")
            if score < 0.0 or score > (max_score * 1.05):
                raise TypeSafeResponseValidationError(
                    f"Score ({score}) outside expected range [0, {max_score}]."
                )
