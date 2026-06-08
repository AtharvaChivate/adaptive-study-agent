from __future__ import annotations

import re
from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from dataclasses import dataclass, field


_ALLOWED_QUESTION_TYPES = {"mcq", "scenario_mcq", "practical_mcq"}
_FORBIDDEN_PHRASES = {
    "all of the above",
    "none of the above",
    "both a and b",
    "all options",
}


class AifQuestion(BaseModel):
    """Pydantic model for a generated AWS AIF practice question."""
    question_id: str = Field(..., description="Stable unique identifier for the question")
    label: str = Field(..., description="UI label, e.g., 'Q1'")
    text: str = Field(..., min_length=12, description="The full question text")
    options: List[str] = Field(..., min_items=4, max_items=4, description="List of 4 unique options")
    correct_option: Literal["A", "B", "C", "D"] = Field(..., description="The correct answer letter")
    explanation: str = Field(..., min_length=1, description="Explanation of why the answer is correct")
    topic_id: str = Field(..., description="AWS Blueprint Task ID (e.g., D1.T1.1)")
    question_type: Literal["mcq", "scenario_mcq", "practical_mcq"] = "mcq"

    @field_validator("topic_id")
    @classmethod
    def validate_topic_id_format(cls, v: str) -> str:
        # Pattern for Domain X, Task Y, Objective Z (e.g., D1.T2.1)
        if not re.match(r"^D\d+\.T\d+\.\d+$", v):
            raise ValueError(f"topic_id '{v}' must be a valid Blueprint Task ID in DX.TX.X format")
        return v

    @field_validator("options")
    @classmethod
    def validate_unique_options(cls, v: List[str]) -> List[str]:
        if len(set(opt.strip().lower() for opt in v)) != 4:
            raise ValueError("All 4 options must be unique and non-empty")
        return v

    @model_validator(mode="after")
    def check_forbidden_phrases_and_alignment(self) -> AifQuestion:
        text_low = self.text.lower()
        expl_low = self.explanation.lower()
        
        for phrase in _FORBIDDEN_PHRASES:
            if phrase in text_low or phrase in expl_low:
                raise ValueError(f"Forbidden phrase detected: '{phrase}'")

        # Ensure explanation mentions the correct answer choice or its letter
        idx = ord(self.correct_option) - ord("A")
        correct_text = self.options[idx].lower()
        if correct_text not in expl_low and self.correct_option.lower() not in expl_low:
            raise ValueError("Explanation must reference the correct answer or its option letter")
            
        return self


class QuestionBatch(BaseModel):
    """Wrapper for a batch of generated questions."""
    questions: List[AifQuestion]


class AifGradedAnswer(BaseModel):
    """Pydantic model for a graded response."""
    question_id: str
    label: str
    user_answer: str
    is_correct: bool
    short_feedback: str = Field(..., description="1-2 sentences of conversational feedback")
    detailed_explanation: str = Field(..., description="Deep dive into the concept")
    topic_id: str
    mastery_delta: float = Field(..., ge=-0.2, le=0.2, description="Adjustment to mastery score")


class GradingBatch(BaseModel):
    """Wrapper for a batch of graded answers."""
    graded: List[AifGradedAnswer]


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
