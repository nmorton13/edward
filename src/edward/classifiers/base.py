"""Base classifier interface and typed request/result domain models."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    """Payload to classify."""

    record_id: str
    object_type: Literal["resource", "capture", "finding"]
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgmentItem(BaseModel):
    """Individual taxonomic judgment produced by a classifier."""

    label_or_question_id: str
    family: Literal["form", "topic", "signal", "custom"]
    primitive: Literal["choice", "noul", "score"]
    answer: dict[str, Any]
    probability: float | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    """Normalized response from any classifier provider."""

    record_id: str
    provider: str
    model: str
    requested_model: str | None = None
    resolved_model: str | None = None
    judgments: list[JudgmentItem] = Field(default_factory=list)
    provider_request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class BaseClassifier(ABC):
    """Abstract interface for all Edward classifier backends."""

    @abstractmethod
    def classify(self, request: ClassificationRequest) -> ClassificationResult:
        """Classify a request and return typed judgments."""
        pass
