"""Wire primitives for System One classification: Choice, Noul, Score."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChoiceQuestion(BaseModel):
    """Closed-set selection for primary form or category."""

    model_config = ConfigDict(extra="allow")

    question_id: str
    instructions: str | None = None
    criteria: list[str] | dict[str, Any] | None = None
    options: list[str] | list[dict[str, Any]] = Field(default_factory=list)
    prompt: str | None = None


class ChoiceAnswer(BaseModel):
    """Choice answer with single selected option and complete probability distribution."""

    model_config = ConfigDict(extra="allow")

    primitive: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None

    @property
    def selected(self) -> str:
        """Compatibility accessor for legacy or internal Edward consumers."""
        return self.choice


class NoulQuestion(BaseModel):
    """Independent binary probability question for overlapping topics or signals."""

    model_config = ConfigDict(extra="allow")

    question_id: str
    instructions: str | None = None
    criteria: list[str] | dict[str, Any] | None = None
    prompt: str | None = None


class NoulAnswer(BaseModel):
    """Noul answer with probability value in [0.0, 1.0]."""

    model_config = ConfigDict(extra="allow")

    primitive: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)
    confidence: float | None = None

    @property
    def probability(self) -> float:
        """Compatibility accessor for probability attribute."""
        return self.noul


class ScoreQuestion(BaseModel):
    """Ordered rubric evaluation for quality, depth, or technical rigor."""

    model_config = ConfigDict(extra="allow")

    question_id: str
    instructions: str | None = None
    criteria: list[str] | dict[str, Any] | None = None
    legend: list[str] | dict[str, Any] | None = None
    prompt: str | None = None


class ScoreAnswer(BaseModel):
    """Score answer on an ordered rubric scale with fractional score support."""

    model_config = ConfigDict(extra="allow")

    primitive: Literal["score"] = "score"
    score: float
    max_score: float = 1.0
    probabilities: dict[str, float] | list[float] | None = None
    legend: list[str] | dict[str, str] | None = None
    confidence: float | None = None
