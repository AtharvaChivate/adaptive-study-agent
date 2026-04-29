from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import os

from groq import Groq


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

    def generate_questions(
        self,
        days_until_exam: int,
        num_questions: int,
        weak_topics: List[str],
        domains_weighting: Dict[str, float],
    ) -> List[GeneratedQuestion]:
        system_prompt = (
            "You are an expert AWS instructor preparing a user for the AWS "
            "Artificial Intelligence Foundations (AIF) exam. Generate high-quality practice "
            "questions that match the official blueprint and domain weightings."
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
- options: array of 4 options (A-D). **Do NOT prefix options with letters**;
    return only the raw option text. The email formatter will add "A.", "B.", etc.
- correct_option: a single letter A-D
- explanation: short explanation of correct answer
- topic_id: a string mapping to domain/task (e.g. "D2.T2.1") or key AWS service name
- question_type: one of ["mcq", "scenario_mcq", "practical_mcq"]
"""

        completion = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )

        content = completion.choices[0].message.content
        import json

        data = json.loads(content)
        questions_raw = data.get("questions", [])
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
