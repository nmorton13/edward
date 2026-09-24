"""Deterministic development classifier operating without external network dependencies."""

import json
import re
from pathlib import Path
from typing import Any

from edward.classifiers.base import (
    BaseClassifier,
    ClassificationRequest,
    ClassificationResult,
    JudgmentItem,
)
from edward.classifiers.system_one import ChoiceAnswer, NoulAnswer


class DryRunClassifier(BaseClassifier):
    """Deterministic classifier that uses packaged registries and keyword matching."""

    def __init__(self, registry_dir: Path | None = None) -> None:
        self.registry_dir = registry_dir or (Path(__file__).parent.parent / "registries")
        self._load_registries()

    def _load_registries(self) -> None:
        def _read_json(name: str) -> dict[str, Any]:
            p = self.registry_dir / name
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
            return {}

        self.forms_reg = _read_json("forms-v1.json")
        self.topics_reg = _read_json("topics-v1.json")
        self.signals_reg = _read_json("signals-v1.json")
        self.questions_reg = _read_json("jev-questions-v1.json")

    def classify(self, request: ClassificationRequest) -> ClassificationResult:
        text_lower = request.text.lower()
        url = str(request.metadata.get("url") or "").lower()
        judgments: list[JudgmentItem] = []

        # 1. Primary Form (Choice)
        form_options = [
            "article",
            "paper",
            "repository",
            "x-post",
            "email-note",
            "personal-note",
            "other",
        ]
        form_scores: dict[str, float] = dict.fromkeys(form_options, 0.1)

        if "x.com" in url or "twitter.com" in url:
            form_scores["x-post"] += 2.0
        elif "github.com" in url or "gitlab.com" in url or "git clone" in text_lower:
            form_scores["repository"] += 2.0
        elif "arxiv.org" in url or "abstract" in text_lower and "references" in text_lower:
            form_scores["paper"] += 2.0
        elif not url and len(text_lower) < 500:
            form_scores["personal-note"] += 1.5
        elif url:
            form_scores["article"] += 1.0
        else:
            form_scores["other"] += 0.5

        # Softmax / Normalize
        total_score = sum(form_scores.values())
        form_probs = {k: round(v / total_score, 4) for k, v in form_scores.items()}
        selected_form = max(form_probs.items(), key=lambda x: x[1])[0]

        choice_ans = ChoiceAnswer(
            choice=selected_form,
            probabilities=form_probs,
            confidence=0.95,
        )
        ans_dict = choice_ans.model_dump()
        ans_dict["selected"] = selected_form
        judgments.append(
            JudgmentItem(
                label_or_question_id="primary-form",
                family="form",
                primitive="choice",
                answer=ans_dict,
                probability=form_probs[selected_form],
                confidence=0.95,
            )
        )

        # 2. Topics (Noul)
        topics = self.topics_reg.get("entries") or self.topics_reg.get("topics", [])
        for t in topics:
            t_id = t["id"]
            keywords = t_id.replace("/", " ").split() + [t_id]
            desc_words = t.get("description", "").lower().split()
            keywords += [w for w in desc_words if len(w) > 4]
            match_count = sum(
                1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text_lower)
            )
            prob = 0.85 if match_count > 0 else 0.05

            noul_ans = NoulAnswer(noul=prob, confidence=0.9)
            n_dict = noul_ans.model_dump()
            n_dict["probability"] = prob
            judgments.append(
                JudgmentItem(
                    label_or_question_id=t_id,
                    family="topic",
                    primitive="noul",
                    answer=n_dict,
                    probability=prob,
                    confidence=0.9,
                )
            )

        # 3. Signals (Noul)
        signals = self.signals_reg.get("entries") or self.signals_reg.get("signals", [])
        for s in signals:
            s_id = s["id"]
            keywords = [s_id, s_id.replace("-", " ")]
            desc_words = s.get("description", "").lower().split()
            keywords += [w for w in desc_words if len(w) > 4]
            match_count = sum(
                1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", text_lower)
            )
            prob = 0.80 if match_count > 0 else 0.05

            noul_ans = NoulAnswer(noul=prob, confidence=0.85)
            s_dict = noul_ans.model_dump()
            s_dict["probability"] = prob
            judgments.append(
                JudgmentItem(
                    label_or_question_id=s_id,
                    family="signal",
                    primitive="noul",
                    answer=s_dict,
                    probability=prob,
                    confidence=0.85,
                )
            )

        return ClassificationResult(
            record_id=request.record_id,
            provider="dry-run",
            model="dry-run-v1",
            judgments=judgments,
            cost=0.0,
            input_tokens=len(request.text.split()),
            output_tokens=len(judgments) * 5,
        )
