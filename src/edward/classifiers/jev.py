"""Jev classification subsystem orchestrator using TypeSafe or OpenRouter transports.

Coordinates Choice (primary form), Noul (topics and signals), and Score questions
against versioned packaged registries, evaluates threshold policies, and produces typed judgments.
"""

import json
import logging
from pathlib import Path
from typing import Any

from edward.classifiers.base import (
    BaseClassifier,
    ClassificationRequest,
    ClassificationResult,
    JudgmentItem,
)
from edward.classifiers.providers.openrouter import OpenRouterProvider
from edward.classifiers.providers.typesafe import TypeSafeProvider
from edward.services.privacy import classify_content_data_class

logger = logging.getLogger(__name__)

# Canonical mapping from registry question ID to label family and target label
QUESTION_LABEL_MAPPING: dict[str, tuple[str, str]] = {
    "topic-ai": ("topic", "ai"),
    "topic-local-ai": ("topic", "ai/local-models"),
    "topic-llm-agents": ("topic", "ai/llm-agents"),
    "topic-coding": ("topic", "software/coding"),
    "topic-dev-tools": ("topic", "software/developer-tools"),
    "topic-apple-silicon": ("topic", "hardware/apple-silicon"),
    "topic-privacy": ("topic", "privacy"),
    "topic-writing": ("topic", "writing"),
    "topic-energy-grid": ("topic", "infrastructure/energy-grid"),
    "topic-econ-policy": ("topic", "economics/policy"),
    "topic-austrian-econ": ("topic", "economics/austrian"),
    "topic-bitcoin": ("topic", "crypto/bitcoin"),
    "topic-systems": ("topic", "software/systems"),
    "topic-gaming": ("topic", "gaming"),
    "topic-philosophy": ("topic", "philosophy"),
    "topic-politics": ("topic", "politics/commentary"),
    "topic-creative-tech": ("topic", "creative-tech"),
    "signal-benchmark": ("signal", "benchmark"),
    "signal-first-hand-experience": ("signal", "first-hand-experience"),
    "signal-tutorial": ("signal", "tutorial"),
    "signal-warning": ("signal", "warning"),
    "signal-field-report": ("signal", "field-report"),
    "signal-argument": ("signal", "argument"),
    "signal-announcement": ("signal", "announcement"),
    "signal-cool-project": ("signal", "cool-project"),
    "signal-data-source": ("signal", "data-source"),
}


def load_packaged_questions() -> list[dict[str, Any]]:
    """Load active questions from packaged jev-questions-v1.json."""
    registry_file = Path(__file__).parent.parent / "registries" / "jev-questions-v1.json"
    if not registry_file.exists():
        return []
    try:
        data = json.loads(registry_file.read_text(encoding="utf-8"))
        return [q for q in data.get("questions", []) if q.get("active", True)]
    except Exception as e:
        logger.warning("Failed to load jev-questions-v1.json: %s", e)
        return []


class JevClassifier(BaseClassifier):
    """Orchestrates Jev classification across TypeSafe or OpenRouter transports."""

    def __init__(
        self,
        transport: TypeSafeProvider | OpenRouterProvider | None = None,
        provider_name: str = "typesafe",
    ):
        if transport is not None:
            self.transport = transport
            self.provider_name = getattr(transport, "provider_name", provider_name)
        elif provider_name == "openrouter":
            self.transport = OpenRouterProvider()
            self.provider_name = "openrouter"
        else:
            self.transport = TypeSafeProvider()
            self.provider_name = "typesafe"

    def classify(self, request: ClassificationRequest) -> ClassificationResult:
        """Evaluate Jev questions against text and produce typed judgments."""
        data_class = classify_content_data_class(
            origin_namespace=request.metadata.get("origin_namespace"),
            canonical_url=request.metadata.get("canonical_url"),
            form=request.metadata.get("form"),
            metadata=request.metadata,
        )

        questions = load_packaged_questions()
        if not questions:
            # Fallback if registry could not be read
            questions = [
                {
                    "id": "primary-form",
                    "primitive": "choice",
                    "options": [
                        {"id": "article", "label": "Article"},
                        {"id": "paper", "label": "Paper"},
                        {"id": "repository", "label": "Repository"},
                        {"id": "x-post", "label": "X post"},
                        {"id": "personal-note", "label": "Personal note"},
                        {"id": "other", "label": "Other"},
                    ],
                }
            ]

        # Dispatch questions to underlying transport
        classifier_text = request.text
        if len(classifier_text) > 24_000:
            classifier_text = (
                classifier_text[:18_000]
                + "\n[Middle of source omitted for classification; full text remains stored.]\n"
                + classifier_text[-6_000:]
            )
        eval_result = self.transport.evaluate_questions(
            text=classifier_text,
            questions=questions,
            data_class=data_class,
        )

        answers = eval_result.get("answers", {})
        judgments: list[JudgmentItem] = []

        for q in questions:
            qid = q["id"]
            primitive = q["primitive"]
            ans = answers.get(qid)
            if not ans:
                continue

            if primitive == "choice":
                selected_choice = ans.get("choice") or "other"
                probs = ans.get("probabilities") or {}
                prob = probs.get(selected_choice)
                conf = float(
                    ans.get("confidence", 1.0) if ans.get("confidence") is not None else 1.0
                )
                judgments.append(
                    JudgmentItem(
                        label_or_question_id=qid,
                        family="form",
                        primitive="choice",
                        answer=ans,
                        probability=prob,
                        confidence=conf,
                        metadata={
                            "selected_choice": selected_choice,
                            "question_version": q.get("version", "1.0"),
                        },
                    )
                )

            elif primitive == "noul":
                val = ans.get("noul")
                if val is None:
                    val = ans.get("probability")
                prob = float(val) if val is not None else 0.0
                conf = float(
                    ans.get("confidence", 1.0) if ans.get("confidence") is not None else 1.0
                )

                family, target_label = QUESTION_LABEL_MAPPING.get(qid, ("custom", qid))
                judgments.append(
                    JudgmentItem(
                        label_or_question_id=qid,
                        family=family,  # type: ignore[arg-type]
                        primitive="noul",
                        answer=ans,
                        probability=prob,
                        confidence=conf,
                        metadata={
                            "target_label": target_label,
                            "question_version": q.get("version", "1.0"),
                        },
                    )
                )

            elif primitive == "score":
                score_val = float(ans.get("score", 0.0))
                conf = float(
                    ans.get("confidence", 1.0) if ans.get("confidence") is not None else 1.0
                )
                judgments.append(
                    JudgmentItem(
                        label_or_question_id=qid,
                        family="custom",
                        primitive="score",
                        answer=ans,
                        probability=score_val,
                        confidence=conf,
                        metadata={
                            "max_score": ans.get("max_score", q.get("max_score", 1.0)),
                            "question_version": q.get("version", "1.0"),
                        },
                    )
                )

        req_model = eval_result.get("requested_model") or getattr(
            self.transport, "model", self.provider_name
        )
        res_model = eval_result.get("resolved_model") or eval_result.get("model") or req_model

        return ClassificationResult(
            record_id=request.record_id,
            provider=eval_result.get("provider", self.provider_name),
            model=res_model,
            requested_model=req_model,
            resolved_model=res_model,
            judgments=judgments,
            provider_request_id=eval_result.get("provider_request_id"),
            input_tokens=eval_result.get("input_tokens", 0),
            output_tokens=eval_result.get("output_tokens", 0),
            cost=eval_result.get("cost", 0.0),
        )
