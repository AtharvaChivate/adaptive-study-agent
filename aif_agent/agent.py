from __future__ import annotations

from datetime import datetime
from typing import List, Dict
import re

from .config import load_config, days_until_exam
from .db import AifDynamoDb
from .llm_client import AifLlmClient, GeneratedQuestion
from .email_client import build_question_email, send_email, format_questions_for_email, parse_reply_body
from .policy import pick_question_count, select_weak_topics, domain_weighting


def _build_db(cfg) -> AifDynamoDb:
    return AifDynamoDb(
        region=cfg.aws_region,
        topic_mastery_table=cfg.dynamodb_topic_mastery_table,
        question_history_table=cfg.dynamodb_question_history_table,
        exam_meta_table=cfg.dynamodb_exam_meta_table,
        daily_question_map_table=cfg.dynamodb_daily_question_map_table,
    )


def _build_llm(cfg) -> AifLlmClient:
    return AifLlmClient(api_key=cfg.groq_api_key, model=cfg.groq_model)


def _label_sort_key(label: str) -> int:
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 0


def _new_batch_id() -> str:
    """Generate a unique, human-readable batch id.

    Format: YYYY-MM-DD-HHMMSS (UTC). This allows multiple sends per day
    without collisions while keeping the id easy to read and type when
    grading manually.
    """
    return datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")


def _derive_user_answers(reply_body: str, labels_in_order: List[str]) -> Dict[str, str]:
    """Robustly derive answers per label from the reply body.

    1) Prefer the structured parser (Q1: B) from parse_reply_body.
    2) If that yields nothing (for example, due to encoding issues), fall
       back to taking the first A-D letter found on each non-empty line and
       mapping them by position to the provided labels.
    """
    parsed = parse_reply_body(reply_body)
    if parsed:
        return parsed

    answers: Dict[str, str] = {}
    letters: List[str] = []
    for line in reply_body.splitlines():
        m = re.search(r"([A-Da-d])", line)
        if m:
            letters.append(m.group(1).upper())

    for label, letter in zip(labels_in_order, letters):
        answers[label] = letter
    return answers


def run_daily_question_send() -> None:
    """Generate questions, store mappings, and email them to the user."""
    cfg = load_config()
    db = _build_db(cfg)
    llm = _build_llm(cfg)

    days_left = days_until_exam(cfg)

    masteries = db.list_topic_mastery(cfg.exam_name)
    weak_topics = select_weak_topics(masteries)
    num_questions = pick_question_count(cfg, days_left)

    questions = llm.generate_questions(
        days_until_exam=days_left,
        num_questions=num_questions,
        weak_topics=weak_topics,
        domains_weighting=domain_weighting(),
    )

    # De-duplicate questions within a single email based on question text,
    # then normalize labels to Q1, Q2, ... so they are always sequential.
    seen_text: set[str] = set()
    unique_questions: List[GeneratedQuestion] = []
    for q in questions:
        key = q.text.strip().lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        unique_questions.append(q)

    for idx, q in enumerate(unique_questions, start=1):
        q.label = f"Q{idx}"

    batch_id = _new_batch_id()
    _persist_question_batch(db, cfg.exam_name, batch_id, unique_questions)

    body = format_questions_for_email([
        {
            "label": q.label,
            "text": q.text,
            "options": q.options,
        }
        for q in unique_questions
    ])

    subject = f"AWS AIF Practice Questions - {batch_id}"
    msg = build_question_email(subject, cfg.user_email, cfg.from_email, body)
    send_email(cfg.smtp_host, cfg.smtp_port, cfg.smtp_username, cfg.smtp_password, msg)


def _persist_question_batch(db: AifDynamoDb, exam_name: str, batch_id: str, questions: List[GeneratedQuestion]) -> None:
    mappings = {q.label: q.question_id for q in questions}
    db.put_daily_question_map(exam_name, batch_id, mappings)

    records = []
    now = datetime.utcnow().isoformat()
    for q in questions:
        records.append(
            {
                "exam_name": exam_name,
                "question_id": q.question_id,
                "batch_id": batch_id,
                "label": q.label,
                "text": q.text,
                "options": q.options,
                "correct_option": q.correct_option,
                "topic_id": q.topic_id,
                "question_type": q.question_type,
                "created_at": now,
                "answered": False,
            }
        )
    db.put_question_history_batch(records)


def grade_from_reply(batch_id: str, reply_body: str) -> None:
    """Grade a reply email body and update mastery + send feedback email.

    For now this assumes the reply body is passed in (for example, pasted
    into a GitHub Actions secret or run manually). Gmail API integration to
    fetch the reply can be added on top of this.
    """
    cfg = load_config()
    db = _build_db(cfg)
    llm = _build_llm(cfg)

    # Fetch original questions for this batch
    mapping = db.get_daily_question_map(cfg.exam_name, batch_id)
    if not mapping:
        raise RuntimeError(f"No daily_question_map found for batch {batch_id}")

    # Fetch stored questions for this batch
    question_items = db.list_questions_for_batch(cfg.exam_name, batch_id)
    if not question_items:
        raise RuntimeError(f"No questions found in question_history for batch {batch_id}")

    # Build lookup by label for convenience
    items_by_label: Dict[str, Dict] = {item["label"]: item for item in question_items}

    # Determine label order
    labels_in_order: List[str] = sorted(items_by_label.keys(), key=_label_sort_key)

    # Derive user answers using robust parsing with fallback
    user_answers = _derive_user_answers(reply_body, labels_in_order)

    # Build GeneratedQuestion list from stored records
    questions: List[GeneratedQuestion] = []
    for item in question_items:
        questions.append(
            GeneratedQuestion(
                question_id=str(item["question_id"]),
                label=str(item["label"]),
                text=str(item["text"]),
                options=[str(o) for o in item.get("options", [])],
                correct_option=str(item.get("correct_option")) if item.get("correct_option") else None,
                explanation=None,
                topic_id=str(item.get("topic_id", "unknown")),
                question_type=str(item.get("question_type", "mcq")),
            )
        )

    # Filter user answers to only known labels
    filtered_answers: Dict[str, str] = {
        label: ans
        for label, ans in user_answers.items()
        if label in items_by_label
    }

    # Grade via LLM
    graded_results = llm.grade_answers(questions, filtered_answers)

    # Update topic mastery using mastery_delta
    masteries = db.list_topic_mastery(cfg.exam_name)
    topic_scores: Dict[str, float] = {m.topic_id: m.score for m in masteries}
    for gr in graded_results:
        current = topic_scores.get(gr.topic_id, 0.5)
        new_score = max(0.0, min(1.0, current + gr.mastery_delta))
        topic_scores[gr.topic_id] = new_score
        db.update_topic_mastery(cfg.exam_name, gr.topic_id, new_score)

    # Persist graded question history
    updated_records = []
    now = datetime.utcnow().isoformat()
    for gr in graded_results:
        item = items_by_label.get(gr.label)
        if not item:
            continue
        record = dict(item)
        record["answered"] = True
        # Prefer the raw parsed answer from the user's reply; fall back
        # to whatever the LLM returned if for some reason parsing failed.
        record["user_answer"] = user_answers.get(gr.label, gr.user_answer)
        record["is_correct"] = gr.is_correct
        record["short_feedback"] = gr.short_feedback
        record["detailed_explanation"] = gr.detailed_explanation
        record["graded_at"] = now
        updated_records.append(record)

    if updated_records:
        db.put_question_history_batch(updated_records)

    # Build feedback email body, sorted by question number
    lines: List[str] = []
    lines.append(f"Feedback for AWS AIF practice questions - batch {batch_id}\n")

    for gr in sorted(graded_results, key=lambda g: _label_sort_key(g.label)):
        item = items_by_label.get(gr.label)
        if not item:
            continue
        status = "Correct" if gr.is_correct else "Incorrect"
        lines.append(f"{gr.label} - {status}")
        lines.append(item.get("text", ""))

        correct_letter = (item.get("correct_option") or "").upper()
        options = item.get("options", [])
        user_ans = user_answers.get(gr.label, gr.user_answer)
        lines.append(f"Your answer: {user_ans}")
        if correct_letter and options:
            letters = ["A", "B", "C", "D"]
            try:
                idx = letters.index(correct_letter)
                correct_text = options[idx]
            except ValueError:
                correct_text = ""
            # Clean any leading label from correct option text, similar to email formatter
            correct_text_clean = re.sub(r"^[A-D][\)\.:\-\]]\s*", "", str(correct_text)).strip()
            suffix = f" ({correct_text_clean})" if correct_text_clean else ""
            lines.append(f"Correct answer: {correct_letter}{suffix}")

        lines.append(f"Why: {gr.short_feedback}")
        lines.append(str(gr.detailed_explanation))
        lines.append("")

    body = "\n".join(lines)
    subject = f"AWS AIF Feedback - {batch_id}"
    msg = build_question_email(subject, cfg.user_email, cfg.from_email, body)
    send_email(cfg.smtp_host, cfg.smtp_port, cfg.smtp_username, cfg.smtp_password, msg)
