"""Local classifier implementation stub for offline / on-device classification."""

from pathlib import Path

from edward.classifiers.base import (
    BaseClassifier,
    ClassificationRequest,
    ClassificationResult,
)
from edward.classifiers.dry_run import DryRunClassifier


class LocalClassifier(BaseClassifier):
    """Local classifier executing rules or local model heuristics."""

    def __init__(self, registry_dir: Path | None = None) -> None:
        self.fallback = DryRunClassifier(registry_dir=registry_dir)

    def classify(self, request: ClassificationRequest) -> ClassificationResult:
        # Currently delegates to deterministic registry-driven dry-run logic
        res = self.fallback.classify(request)
        return ClassificationResult(
            record_id=request.record_id,
            provider="local",
            model="local-rules-v1",
            judgments=res.judgments,
            cost=0.0,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
        )
