from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from typing import Any, Dict, List
import re
import uuid

import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from .guardrails import GuardrailIssue, GuardrailReport, QuestionGuardrails, QuestionBatch, GradingBatch

logger = logging.getLogger(__name__)
from datetime import datetime


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
    def __init__(self, api_key: str, model: str, db=None, exam_name: str | None = None) -> None:
        if not api_key:
            raise RuntimeError("GROQ_API_KEY must be set")
        self._llm = ChatGroq(
            api_key=api_key, 
            model_name=model, 
            temperature=0.3,
            max_tokens=4096  # Ensure there is enough space for large batches
        )
        self._model = model
        self._db = db
        self._exam_name = exam_name

    def _question_signature(self, question: Dict[str, Any]) -> str:
        """Build a stable dedupe key from question text + options."""

        text = str(question.get("text", "")).strip().lower()
        text = re.sub(r"\s+", " ", text)
        options = question.get("options", [])
        options_norm = [re.sub(r"\s+", " ", str(opt).strip().lower()) for opt in options]
        return f"{text}||{'|'.join(options_norm)}"

    def _token_set(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        stop_words = {
            "the", "and", "for", "with", "that", "from", "this", "what", "which", "when",
            "into", "your", "would", "should", "using", "about", "after", "before", "under",
            "into", "then", "than", "best", "most", "least", "into", "on", "in", "at", "to",
        }
        return {t for t in tokens if len(t) > 2 and t not in stop_words}

    def _jaccard_similarity(self, left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        inter = len(left & right)
        union = len(left | right)
        return inter / union if union else 0.0

    def _is_near_duplicate(
        self,
        candidate: Dict[str, Any],
        accepted_questions: List[Dict[str, Any]],
    ) -> tuple[bool, str]:
        """Reject close variants to increase repetition penalty."""

        candidate_text_tokens = self._token_set(str(candidate.get("text", "")))
        candidate_options_text = " ".join(str(o) for o in candidate.get("options", []))
        candidate_option_tokens = self._token_set(candidate_options_text)
        candidate_topic = str(candidate.get("topic_id", "unknown")).strip().lower()

        for existing in accepted_questions:
            existing_text_tokens = self._token_set(str(existing.get("text", "")))
            existing_options_text = " ".join(str(o) for o in existing.get("options", []))
            existing_option_tokens = self._token_set(existing_options_text)
            existing_topic = str(existing.get("topic_id", "unknown")).strip().lower()

            text_similarity = self._jaccard_similarity(candidate_text_tokens, existing_text_tokens)
            option_similarity = self._jaccard_similarity(candidate_option_tokens, existing_option_tokens)
            same_topic = candidate_topic == existing_topic

            if text_similarity >= 0.78:
                return True, (
                    f"near-duplicate wording detected (text similarity={text_similarity:.2f})"
                )
            if same_topic and text_similarity >= 0.65 and option_similarity >= 0.70:
                return True, (
                    "near-duplicate scenario/options detected in same topic "
                    f"(text={text_similarity:.2f}, options={option_similarity:.2f})"
                )

        return False, ""

    def _generate_once(
        self,
        days_until_exam: int,
        num_questions: int,
        weak_topics: List[str],
        domains_weighting: Dict[str, float],
        allowed_scope_services: List[str] | None = None,
        extra_feedback: str | None = None,
    ) -> GuardrailReport:
        # Issue 8: Guard against empty personalization data
        effective_weak_topics = weak_topics if weak_topics else ["Bedrock", "SageMaker", "Responsible AI"]
        effective_weights = domains_weighting if domains_weighting else {"D1": 0.2, "D2": 0.24, "D3": 0.28, "D4": 0.14, "D5": 0.14}

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert AWS instructor preparing a user for the AWS "
                "Artificial Intelligence Foundations (AIF) exam. Generate high-quality practice "
                "questions that match the official blueprint and domain weightings.\n\n"
                "CONTENT GUIDELINES:\n"
                "- Focus on AI-native services: Amazon Bedrock, SageMaker, A2I, Comprehend, and Rekognition.\n"
                "- 60% of questions must be Scenario-based MCQ (applying knowledge to a business problem).\n"
                "- Ensure options are distinct and plausible but factually incorrect except for the answer.\n\n"
                "STRUCTURAL CRITICAL:\n"
                "1. correct_option MUST be exactly one uppercase letter: 'A', 'B', 'C', or 'D'.\n"
                "   - 'A' represents the first option, 'B' the second, etc.\n"
                "   - DO NOT provide the full text of the answer in this field.\n"
                "2. Stick to factual AWS service capabilities.\n"
                "3. Each question must be meaningfully distinct.\n"
                "4. You MUST include 'question_type' for every question.\n"
                "5. topic_id MUST be a task-level Blueprint ID (e.g., 'D2.T1.2'). Do NOT use service names or top-level Domain IDs like 'D2'.\n"
            )),
            ("user", (
                "The exam is in {days_until_exam} days. Generate {num_questions} questions.\n\n"
                "Constraints:\n"
                "- Focus on weaker topics: {effective_weak_topics}\n"
                "- Respect domain weightings: {effective_weights}\n"
                "- In-scope services: {scope_services}\n"
                "{feedback_section}"
            ))
        ])

        # Issue 3: Implement automatic retry for structured output validation failures
        # issue 1: Add stronger instructions in the prompt
        chain = prompt | self._llm.with_structured_output(QuestionBatch)

        current_extra_feedback = extra_feedback or ""

        # Simple retry logic for schema validation
        for attempt in range(3):
            try:
                feedback_text = f"\nFeedback from previous attempt:\n{current_extra_feedback}" if current_extra_feedback else ""
                response = chain.invoke({
                    "days_until_exam": days_until_exam,
                    "num_questions": num_questions,
                    "effective_weak_topics": effective_weak_topics,
                    "effective_weights": effective_weights,
                    "scope_services": allowed_scope_services or "Standard AIF scope",
                    "feedback_section": feedback_text
                })
                questions_raw = [q.model_dump() for q in response.questions]
                return QuestionGuardrails().validate_questions(questions_raw)
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to generate valid schema after 3 attempts: {e}")
                    raise
                logger.warning(f"Schema validation failed on attempt {attempt+1}, retrying...")
                # Feed the specific structural error back to the model for the next retry attempt
                current_extra_feedback = f"{current_extra_feedback}\n- Structural Error: {str(e)}"

        return GuardrailReport()

    def generate_questions(
        self,
        days_until_exam: int,
        num_questions: int,
        weak_topics: List[str],
        domains_weighting: Dict[str, float],
        allowed_scope_services: List[str] | None = None,
    ) -> List[GeneratedQuestion]:
        max_attempts = 8
        max_consecutive_no_progress = 2
        report = GuardrailReport()
        accepted_questions: List[Dict[str, Any]] = []
        seen_signatures: set[str] = set()
        consecutive_no_progress = 0
        no_progress_hint = False
        retry_feedback_lines: List[str] = []

        for attempt in range(1, max_attempts + 1):
            remaining = max(num_questions - len(accepted_questions), 0)
            if remaining == 0:
                break

            extra_feedback = None
            if retry_feedback_lines:
                extra_feedback = "\n".join(retry_feedback_lines)
            if no_progress_hint:
                additional_hint = (
                    "Generate fresh, non-duplicate questions with distinct scenarios/services "
                    "from previous outputs."
                )
                extra_feedback = f"{extra_feedback}\n- {additional_hint}" if extra_feedback else additional_hint

            report = self._generate_once(
                days_until_exam=days_until_exam,
                num_questions=remaining,
                weak_topics=weak_topics,
                domains_weighting=domains_weighting,
                allowed_scope_services=allowed_scope_services,
                extra_feedback=extra_feedback,
            )

            before_count = len(accepted_questions)
            duplicate_issues: List[GuardrailIssue] = []
            for q in report.valid_questions:
                signature = self._question_signature(q)
                if signature in seen_signatures:
                    duplicate_issues.append(
                        GuardrailIssue(
                            question_id=str(q.get("question_id", "unknown")),
                            label=str(q.get("label", "unknown")),
                            reason="exact duplicate question detected",
                        )
                    )
                    continue

                near_dup, near_dup_reason = self._is_near_duplicate(q, accepted_questions)
                if near_dup:
                    duplicate_issues.append(
                        GuardrailIssue(
                            question_id=str(q.get("question_id", "unknown")),
                            label=str(q.get("label", "unknown")),
                            reason=near_dup_reason,
                        )
                    )
                    continue

                seen_signatures.add(signature)
                accepted_questions.append(q)
                if len(accepted_questions) >= num_questions:
                    break

            retry_feedback_lines = [
                f"- {issue.label} ({issue.question_id}): {issue.reason}"
                for issue in report.rejected_questions
            ]
            retry_feedback_lines.extend(
                f"- {issue.label} ({issue.question_id}): {issue.reason}"
                for issue in duplicate_issues[:8]
            )

            added_this_attempt = len(accepted_questions) - before_count
            if added_this_attempt == 0:
                consecutive_no_progress += 1
                no_progress_hint = True
            else:
                consecutive_no_progress = 0
                no_progress_hint = False

            # Store rejected questions for audit trail
            if self._db and self._exam_name and report.rejected_questions:
                batch_id = f"gen-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                rejected_data = [
                    {
                        "question_id": issue.question_id,
                        "label": issue.label,
                        "reason": issue.reason,
                        "raw_question": {},  # Would ideally have the raw data
                    }
                    for issue in report.rejected_questions
                ]
                self._db.put_rejected_questions_batch(
                    exam_name=self._exam_name,
                    batch_id=batch_id,
                    attempt=attempt,
                    rejected_issues=rejected_data,
                )

            logger.warning(
                "Attempt %s: accepted %s/%s total (added %s, rejected %s, duplicates %s)",
                attempt,
                len(accepted_questions),
                num_questions,
                added_this_attempt,
                report.rejected_count,
                len(duplicate_issues),
            )

            if len(accepted_questions) >= num_questions:
                break

            if consecutive_no_progress >= max_consecutive_no_progress:
                logger.warning(
                    "Stopping early after %s consecutive no-progress attempts",
                    consecutive_no_progress,
                )
                break

        if len(accepted_questions) < num_questions:
            logger.warning(
                "Returning %s valid unique questions after %s attempts; requested %s",
                len(accepted_questions),
                max_attempts,
                num_questions,
            )

        questions: List[GeneratedQuestion] = []
        seen_question_ids: set[str] = set()
        for q in accepted_questions[:num_questions]:
            question_id = str(q["question_id"])
            if question_id in seen_question_ids:
                question_id = f"{question_id}-{uuid.uuid4().hex[:8]}"
                logger.warning("Duplicate question_id detected; assigned unique suffix")
            seen_question_ids.add(question_id)

            # Deterministic check: Ensure correct_option letter is within bounds of options list
            raw_correct = str(q.get("correct_option", "")).upper()
            options = [str(o) for o in q.get("options", [])]
            if raw_correct in ["A", "B", "C", "D"]:
                idx = ord(raw_correct) - ord('A')
                if idx >= len(options):
                    logger.warning(f"correct_option {raw_correct} out of bounds for {len(options)} options in {question_id}. Defaulting to None.")
                    raw_correct = None

            questions.append(
                GeneratedQuestion(
                    question_id=question_id,
                    label=str(q["label"]),
                    text=str(q["text"]),
                    options=options,
                    correct_option=raw_correct if raw_correct else None,
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
        if not questions:
            return []

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an AWS AIF tutor providing feedback on practice questions. "
                "The correctness of the user's answer has already been determined. "
                "Your task is to provide conversational feedback, a detailed explanation of the concept, "
                "and a mastery delta (-0.2 to +0.2) based on the provided results.\n\n"
                "GUIDELINES:\n"
                "- If is_correct is True, provide positive feedback.\n"
                "- If is_correct is False, explain the mistake and reinforce why the correct_option is right.\n"
                "- Your explanation MUST remain consistent with the provided correct_option."
            )),
            ("user", "Provide feedback for the following results:\n\n{payload}")
        ])

        truth_map: Dict[str, bool] = {}
        payload = []
        for q in questions:
            user_ans = user_answers.get(q.label, "").upper()
            correct_ans = (q.correct_option or "").upper()
            is_correct = user_ans == correct_ans if user_ans else False
            truth_map[q.label] = is_correct

            payload.append(
                {
                    "label": q.label,
                    "question_id": q.question_id,
                    "text": q.text,
                    "options": q.options,
                    "correct_option": q.correct_option,
                    "topic_id": q.topic_id,
                    "user_answer": user_ans,
                    "is_correct": is_correct,
                }
            )

        chain = prompt | self._llm.with_structured_output(GradingBatch)
        
        # Handle potential API truncation (400 errors) with retries
        for attempt in range(3):
            try:
                response = chain.invoke({
                    "payload": payload
                })

                results: List[GradedAnswer] = []
                for g in response.graded:
                    # Get original question for consistency checking
                    orig_q = next((q for q in questions if q.label == g.label), None)

                    detailed = g.detailed_explanation
                    if isinstance(detailed, list):
                        detailed_str = "\n".join(f"- {str(line)}" for line in detailed)
                    else:
                        detailed_str = str(detailed)

                    # Enforce deterministic truth and delta signs
                    is_correct = truth_map.get(g.label, g.is_correct)

                    # Consistency check: Ensure explanation aligns with ground truth correct_option
                    if orig_q and orig_q.correct_option:
                        idx = ord(orig_q.correct_option.upper()) - ord('A')
                        correct_text = orig_q.options[idx].lower()
                        detailed_low = detailed_str.lower()
                        
                        # If explanation doesn't mention the correct answer text or letter, it might be drifting
                        if orig_q.correct_option.lower() not in detailed_low and correct_text not in detailed_low:
                            logger.warning(f"Graded explanation for {g.label} appears inconsistent with answer key. Appending truth.")
                            detailed_str += f"\n\n[System Note]: The correct answer is {orig_q.correct_option}: {orig_q.options[idx]}."
                            if orig_q.explanation:
                                detailed_str += f" {orig_q.explanation}"

                    m_delta = g.mastery_delta
                    if is_correct and m_delta < 0:
                        m_delta = abs(m_delta) if m_delta != 0 else 0.05
                    elif not is_correct and m_delta > 0:
                        m_delta = -abs(m_delta) if m_delta != 0 else -0.05

                    results.append(
                        GradedAnswer(
                            question_id=g.question_id,
                            label=g.label,
                            user_answer=g.user_answer,
                            is_correct=is_correct,
                            short_feedback=g.short_feedback,
                            detailed_explanation=detailed_str,
                            topic_id=g.topic_id,
                            mastery_delta=m_delta,
                        )
                    )
                return results
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to grade answers after 3 attempts: {e}")
                    raise
                logger.warning(f"Grading failed on attempt {attempt+1}, retrying...")
        
        return []
