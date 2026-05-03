from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


_ALLOWED_QUESTION_TYPES = {"mcq", "scenario_mcq", "practical_mcq"}
_FORBIDDEN_PHRASES = {
    "all of the above",
    "none of the above",
    "both a and b",
    "all options",
}


@dataclass
class GuardrailIssue:
    question_id: str
    label: str
    reason: str


@dataclass
class GuardrailReport:
    valid_questions: List[Dict[str, Any]] = field(default_factory=list)
    rejected_questions: List[GuardrailIssue] = field(default_factory=list)

    @property
    def valid_count(self) -> int:
        return len(self.valid_questions)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_questions)


class QuestionGuardrails:
    """Deterministic guardrails for generated practice questions.

    These checks are intentionally conservative. They reject questions that are
    structurally invalid or likely to be misleading before they can reach the UI.
    """

    def validate_question(self, question: Dict[str, Any]) -> Tuple[bool, str]:
        question_id = str(question.get("question_id", "unknown"))
        label = str(question.get("label", "unknown"))
        text = str(question.get("text", "")).strip()
        explanation = str(question.get("explanation", "")).strip()
        question_type = str(question.get("question_type", "mcq")).strip().lower()
        correct_option = str(question.get("correct_option", "")).strip().upper()
        options = question.get("options", [])

        if not text:
            return False, "question text is empty"
        if len(text) < 12:
            return False, "question text is too short"
        if question_type not in _ALLOWED_QUESTION_TYPES:
            return False, f"unsupported question_type '{question_type}'"
        if not explanation:
            return False, "missing explanation"
        if correct_option not in {"A", "B", "C", "D"}:
            return False, f"correct_option '{correct_option}' is not A-D"
        if not isinstance(options, list):
            return False, "options must be a list"
        if len(options) != 4:
            return False, f"expected 4 options, got {len(options)}"

        normalized_options = [str(option).strip() for option in options]
        if any(not option for option in normalized_options):
            return False, "one or more options are empty"
        if len(set(option.lower() for option in normalized_options)) != 4:
            return False, "options must be unique"

        index = ord(correct_option) - ord("A")
        try:
            correct_text = normalized_options[index]
        except IndexError:
            return False, f"correct_option '{correct_option}' maps outside the options list"

        if not correct_text:
            return False, f"correct option '{correct_option}' is empty"

        text_lower = text.lower()
        explanation_lower = explanation.lower()
        for phrase in _FORBIDDEN_PHRASES:
            if phrase in text_lower:
                return False, f"forbidden phrase in question text: '{phrase}'"
            if phrase in explanation_lower:
                return False, f"forbidden phrase in explanation: '{phrase}'"

        # Require the explanation to reference the chosen answer or its meaning.
        if correct_text.lower() not in explanation_lower and correct_option.lower() not in explanation_lower:
            return False, "explanation does not reference the chosen correct answer"

        return True, ""

    def validate_questions(self, questions: List[Dict[str, Any]]) -> GuardrailReport:
        report = GuardrailReport()
        for question in questions:
            valid, reason = self.validate_question(question)
            if valid:
                report.valid_questions.append(question)
                continue

            report.rejected_questions.append(
                GuardrailIssue(
                    question_id=str(question.get("question_id", "unknown")),
                    label=str(question.get("label", "unknown")),
                    reason=reason,
                )
            )
        return report
