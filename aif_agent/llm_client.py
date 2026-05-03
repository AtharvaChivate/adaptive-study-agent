from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import os

from groq import Groq

from .guardrails import GuardrailReport, QuestionGuardrails

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQuestion:
    question_id: str
    label: str  # e.g. "Q1"
    text: str
    options: List[str]
    correct_option: str | None  # optional, can be hidden from email
    explanation: str | None
    topic_id: str  # maps to exam domain / objective
    question_type: str  # mcq | scenario_mcq | practical_mcq


@dataclass
class GradedAnswer:
    question_id: str
    label: str
    user_answer: str
    is_correct: bool
    short_feedback: str
    detailed_explanation: str
    topic_id: str
    mastery_delta: float


class AifLlmClient:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise RuntimeError("GROQ_API_KEY must be set")
        self._client = Groq(api_key=api_key)
        self._model = model

    def _parse_generated_questions(self, content: str) -> List[Dict[str, Any]]:
        data = json.loads(content)
        questions_raw = data.get("questions", [])
        return questions_raw if isinstance(questions_raw, list) else []

    def _generate_once(
        self,
        days_until_exam: int,
        num_questions: int,
        weak_topics: List[str],
        domains_weighting: Dict[str, float],
        extra_feedback: str | None = None,
    ) -> GuardrailReport:
        system_prompt = (
            "You are an expert AWS instructor preparing a user for the AWS "
            "Artificial Intelligence Foundations (AIF) exam. Generate high-quality practice "
            "questions that match the official blueprint and domain weightings.\n\n"
            "CRITICAL: For each question you generate:\n"
            "1. The correct_option MUST be a letter A-D that corresponds to the correct answer in the options list.\n"
            "2. The correct answer MUST exist in the options array.\n"
            "3. Verify that your chosen correct_option actually matches one of the 4 options.\n"
            "4. Never generate a question where the correct answer is not in the options.\n"
            "5. Stick to factual AWS service capabilities. Do not mix incompatible services.\n"
            "6. Avoid questions that use unsupported services or that omit the correct answer from the options."
        )

        user_prompt = f"""
The exam is in {days_until_exam} days.
Generate {num_questions} questions to send by email.

Question types mix:
- Multiple choice (concept checks)
- Scenario-based MCQ
- Case Study Based MCQ
- Real-life practical decision MCQ (what should you do next / which service fits best?)

Constraints:
- Focus more on weaker topics: {weak_topics}.
- Respect domain weightings: {domains_weighting}.
- Difficulty should ramp up slightly as exam day approaches.

For each question, return JSON only, as a list under key `questions`.
Each item must be an object with keys:
- question_id: stable id, string
- label: like "Q1", "Q2", ...
- text: full question text
- options: array of exactly 4 options (A-D). **Do NOT prefix options with letters**;
    return only the raw option text. The email formatter will add "A.", "B.", etc.
- correct_option: a single letter A-D that MUST correspond to the correct answer in the options array
- explanation: short explanation of correct answer, verifying it's actually correct
- topic_id: a string mapping to domain/task (e.g. "D2.T2.1") or key AWS service name
- question_type: one of ["mcq", "scenario_mcq", "practical_mcq"]

IMPORTANT: Before finalizing each question, verify:
- correct_option is valid (A, B, C, or D)
- correct_option index points to the correct answer in the options list
- The explanation confirms why that option is correct
"""

        if extra_feedback:
            user_prompt += f"""

Rejected question feedback from previous attempt:
{extra_feedback}

Fix the issues above and regenerate only valid questions.
"""

        completion = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        questions_raw = self._parse_generated_questions(completion.choices[0].message.content)
        return QuestionGuardrails().validate_questions(questions_raw)

    def generate_questions(
        self,
        days_until_exam: int,
        num_questions: int,
        weak_topics: List[str],
        domains_weighting: Dict[str, float],
    ) -> List[GeneratedQuestion]:
        max_attempts = 3
        report = GuardrailReport()
        questions_raw: List[Dict[str, Any]] = []

        for attempt in range(1, max_attempts + 1):
            extra_feedback = None
            if report.rejected_questions:
                extra_feedback = "\n".join(
                    f"- {issue.label} ({issue.question_id}): {issue.reason}"
                    for issue in report.rejected_questions
                )

            report = self._generate_once(
                days_until_exam=days_until_exam,
                num_questions=num_questions,
                weak_topics=weak_topics,
                domains_weighting=domains_weighting,
                extra_feedback=extra_feedback,
            )
            questions_raw = report.valid_questions

            if len(questions_raw) >= num_questions:
                if report.rejected_count:
                    logger.warning(
                        "Guardrails filtered out %s invalid questions on attempt %s",
                        report.rejected_count,
                        attempt,
                    )
                break

            logger.warning(
                "Guardrails returned %s/%s valid questions on attempt %s",
                report.valid_count,
                num_questions,
                attempt,
            )

        if len(questions_raw) < num_questions:
            logger.warning(
                "Returning %s valid questions after %s attempts; requested %s",
                len(questions_raw),
                max_attempts,
                num_questions,
            )

        questions: List[GeneratedQuestion] = []
        for q in questions_raw:
            questions.append(
                GeneratedQuestion(
                    question_id=str(q["question_id"]),
                    label=str(q["label"]),
                    text=str(q["text"]),
                    options=[str(o) for o in q["options"]],
                    correct_option=str(q.get("correct_option")) if q.get("correct_option") else None,
                    explanation=str(q.get("explanation")) if q.get("explanation") else None,
                    topic_id=str(q.get("topic_id", "unknown")),
                    question_type=str(q.get("question_type", "mcq")),
                )
            )
        return questions

    def grade_answers(
        self,
        questions: List[GeneratedQuestion],
        user_answers: Dict[str, str],  # label -> option letter
    ) -> List[GradedAnswer]:
        system_prompt = (
            "You are grading AWS AIF practice questions. For each question, "
            "decide if the user's answer is correct, explain why, and provide "
            "a mastery delta between -0.2 and +0.2."
        )

        payload = []
        for q in questions:
            payload.append(
                {
                    "label": q.label,
                    "question_id": q.question_id,
                    "text": q.text,
                    "options": q.options,
                    "correct_option": q.correct_option,
                    "topic_id": q.topic_id,
                    "user_answer": user_answers.get(q.label),
                }
            )

        user_prompt = f"""
You are given practice questions with the model answer and the user's answer.
For each, grade and explain.

Return JSON with key `graded`, a list where each item has:
- question_id
- label
- user_answer
- is_correct (bool)
- short_feedback (1-2 sentences, conversational)
- detailed_explanation (3-6 bullet-style sentences, teach the concept)
- topic_id
- mastery_delta (float between -0.2 and 0.2)

Questions:
{payload}
"""

        completion = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        import json

        content = completion.choices[0].message.content
        data = json.loads(content)
        graded_raw = data.get("graded", [])
        results: List[GradedAnswer] = []
        for g in graded_raw:
            detailed = g.get("detailed_explanation", "")
            if isinstance(detailed, list):
                detailed_str = "\n".join(f"- {str(line)}" for line in detailed)
            else:
                detailed_str = str(detailed)
            results.append(
                GradedAnswer(
                    question_id=str(g["question_id"]),
                    label=str(g["label"]),
                    user_answer=str(g.get("user_answer", "")),
                    is_correct=bool(g.get("is_correct", False)),
                    short_feedback=str(g.get("short_feedback", "")),
                    detailed_explanation=detailed_str,
                    topic_id=str(g.get("topic_id", "unknown")),
                    mastery_delta=float(g.get("mastery_delta", 0.0)),
                )
            )
        return results
