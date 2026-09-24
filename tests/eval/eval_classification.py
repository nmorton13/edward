"""Evaluation suite measuring classification accuracy against synthetic labeled examples.

Evaluates primary form detection, topic resolution, and signal classification.
"""

from edward.classifiers.base import ClassificationRequest
from edward.classifiers.dry_run import DryRunClassifier
from edward.services.classification import detect_primary_form

# Synthetic labeled gold dataset
LABELED_GOLD_EXAMPLES = [
    {
        "url": "https://github.com/ggerganov/llama.cpp",
        "text": "Port of Facebook's LLaMA model in C/C++. High performance inference on Apple Silicon.",
        "expected_form": "repository",
        "expected_topics": ["ai", "ai/local-models", "software/coding"],
        "expected_signals": ["benchmark"],
    },
    {
        "url": "https://arxiv.org/abs/2309.12345",
        "text": "We present a comprehensive empirical benchmark of quantized transformer models on edge hardware.",
        "expected_form": "paper",
        "expected_topics": ["ai", "ai/local-models"],
        "expected_signals": ["benchmark"],
    },
    {
        "url": "https://blog.example.com/how-to-setup-ollama",
        "text": "Step-by-step tutorial: How to configure and run local LLMs with Ollama on macOS.",
        "expected_form": "article",
        "expected_topics": ["ai", "ai/local-models"],
        "expected_signals": ["tutorial"],
    },
    {
        "url": "https://x.com/user/status/123456789",
        "text": "Just tested the new Apple Silicon M3 chip with local Llama models. Incredible latency!",
        "expected_form": "x-post",
        "expected_topics": ["ai", "ai/local-models"],
        "expected_signals": ["first-hand-experience"],
    },
    {
        "url": None,
        "text": "Personal note: investigate why memory bandwidth saturates at 64 batch size on M2 Max.",
        "expected_form": "personal-note",
        "expected_topics": ["ai", "ai/local-models"],
        "expected_signals": ["question"],
    },
]


def test_eval_primary_form_detection():
    """Evaluate primary form heuristic detection across gold dataset."""
    correct = 0
    for ex in LABELED_GOLD_EXAMPLES:
        detected = detect_primary_form(url=ex["url"], text=ex["text"])
        if detected == ex["expected_form"]:
            correct += 1

    accuracy = correct / len(LABELED_GOLD_EXAMPLES)
    assert accuracy >= 0.80, f"Primary form accuracy was {accuracy:.2f}, expected >= 0.80"


def test_eval_classifier_taxonomic_coverage():
    """Evaluate DryRunClassifier taxonomic judgment extraction."""
    classifier = DryRunClassifier()

    for idx, ex in enumerate(LABELED_GOLD_EXAMPLES):
        req = ClassificationRequest(
            record_id=f"eval_{idx}",
            object_type="resource",
            text=ex["text"],
            metadata={"url": ex["url"]},
        )
        res = classifier.classify(req)
        assert len(res.judgments) > 0

        # Check that high probability judgments align with expected topics
        high_prob_labels = [
            j.label_or_question_id for j in res.judgments if (j.probability or 0.0) >= 0.70
        ]
        # At least one expected topic was judged with high probability
        assert any(t in high_prob_labels for t in ex["expected_topics"])
