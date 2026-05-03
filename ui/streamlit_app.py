from __future__ import annotations

import os
import sys
import time
import uuid
import logging
from typing import List, Dict
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Setup debug logging
log_file = Path(__file__).parent / "debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Add workspace root to sys.path so imports work when running from ui/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from aif_agent.llm_client import GeneratedQuestion, GradedAnswer, AifLlmClient
from aif_agent.db import AifDynamoDb
from aif_agent.config import load_config, days_until_exam


def mock_generate_questions(num_questions: int) -> List[GeneratedQuestion]:
    questions: List[GeneratedQuestion] = []
    for i in range(1, num_questions + 1):
        qid = str(uuid.uuid4())
        label = f"Q{i}"
        text = f"Sample question {i}: Which option is best for scenario {i}?"
        options = [f"Option {c} for {i}" for c in ["A", "B", "C", "D"]]
        correct = ["A", "B", "C", "D"][(i - 1) % 4]
        questions.append(
            GeneratedQuestion(
                question_id=qid,
                label=label,
                text=text,
                options=options,
                correct_option=correct,
                explanation=f"Because option {correct} is correct.",
                topic_id="mock.topic",
                question_type="mcq",
            )
        )
    return questions


def try_build_llm_and_generate(num_questions: int) -> List[GeneratedQuestion]:
    # Attempt to use real LLM when GROQ_API_KEY is set, otherwise fall back to mock.
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    logger.info(f"GROQ_API_KEY set: {bool(groq_key)}")
    
    if not groq_key:
        logger.info("No GROQ_API_KEY found. Using mock questions.")
        st.info("No GROQ_API_KEY found. Using mock questions.")
        return mock_generate_questions(num_questions)
    
    # Get model from env or default
    groq_model = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    
    # Try to get days_to_exam from config or env; default to 30
    days_to_exam = 30
    try:
        days_override = os.getenv("AIF_DAYS_TO_EXAM")
        if days_override:
            days_to_exam = int(days_override)
        else:
            # Try config
            cfg = load_config()
            days_to_exam = days_until_exam(cfg)
            logger.info(f"Config loaded: days to exam = {days_to_exam}")
    except Exception as e:
        logger.warning(f"Could not determine days to exam (using default 30): {e}")
        days_to_exam = 30

    try:
        logger.info(f"Building LLM client with model {groq_model}...")
        llm = AifLlmClient(api_key=groq_key, model=groq_model)
        logger.info(f"Generating {num_questions} questions (days to exam: {days_to_exam})...")
        questions = llm.generate_questions(
            days_until_exam=days_to_exam, 
            num_questions=num_questions, 
            weak_topics=[], 
            domains_weighting={}
        )
        logger.info(f"Successfully generated {len(questions)} real questions via LLM")
        st.success(f"✅ Generated {len(questions)} real questions via LLM")
        return questions
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        st.warning(f"LLM generation failed: {e}. Using mock questions.")
        return mock_generate_questions(num_questions)


def _load_dynamodb_settings() -> dict[str, str]:
    """Load only the settings needed to write to DynamoDB.

    This intentionally does not require exam-date configuration, so local UI
    persistence can work independently from question-generation timing.
    """
    return {
        "exam_name": os.getenv("AIF_EXAM_NAME", "AWS AIF"),
        "aws_region": os.getenv("AWS_REGION", "us-east-1"),
        "topic_mastery_table": os.getenv("AIF_TOPIC_MASTERY_TABLE", "aif_topic_mastery"),
        "question_history_table": os.getenv("AIF_QUESTION_HISTORY_TABLE", "aif_question_history"),
        "exam_meta_table": os.getenv("AIF_EXAM_META_TABLE", "aif_exam_meta"),
        "daily_question_map_table": os.getenv("AIF_DAILY_QUESTION_MAP_TABLE", "aif_daily_question_map"),
    }


def persist_batch_if_possible(questions: List[GeneratedQuestion], batch_id: str) -> bool:
    cfg = _load_dynamodb_settings()
    logger.info(f"Loaded DynamoDB settings for exam: {cfg['exam_name']}")

    try:
        logger.info("Building DynamoDB client...")
        db = AifDynamoDb(
            region=cfg["aws_region"],
            topic_mastery_table=cfg["topic_mastery_table"],
            question_history_table=cfg["question_history_table"],
            exam_meta_table=cfg["exam_meta_table"],
            daily_question_map_table=cfg["daily_question_map_table"],
        )
    except Exception as e:
        logger.error(f"Failed to build DynamoDB client: {e}", exc_info=True)
        st.warning(f"Failed to build DynamoDB client: {e}")
        return False

    # mappings and records compatible with existing backend
    mappings = {q.label: q.question_id for q in questions}
    try:
        logger.info(f"Writing daily question map for batch {batch_id}...")
        db.put_daily_question_map(cfg["exam_name"], batch_id, mappings)
    except Exception as e:
        logger.warning(f"Failed to write daily question map: {e}", exc_info=True)
        st.warning(f"Failed to write daily question map: {e}")
    records = []
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for q in questions:
        records.append(
            {
                "exam_name": cfg["exam_name"],
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
    try:
        logger.info(f"Writing {len(records)} question history records...")
        db.put_question_history_batch(records)
        logger.info("Batch persisted successfully to DynamoDB")
    except Exception as e:
        logger.error(f"Failed to write question history: {e}", exc_info=True)
        st.warning(f"Failed to write question history: {e}")
        return False
    return True


def grade_locally(questions: List[GeneratedQuestion], answers: Dict[str, str]) -> List[GradedAnswer]:
    graded: List[GradedAnswer] = []
    for q in questions:
        ua = answers.get(q.label, "")
        is_correct = (ua.upper() == (q.correct_option or "").upper()) if ua else False
        short = "Correct" if is_correct else "Incorrect"
        detailed = f"Answer: {ua}. Explanation: {q.explanation or 'No explanation.'}"
        graded.append(
            GradedAnswer(
                question_id=q.question_id,
                label=q.label,
                user_answer=ua,
                is_correct=is_correct,
                short_feedback=short,
                detailed_explanation=detailed,
                topic_id=q.topic_id,
                mastery_delta=0.0,
            )
        )
    return graded


def main() -> None:
    st.set_page_config(page_title="AIF Practice Test", layout="centered")
    st.title("AIF — Practice Test")

    if "state" not in st.session_state:
        st.session_state.state = "landing"
        st.session_state.flagged_questions = set()

    if st.session_state.state == "landing":
        st.write("A lightweight interface to run practice tests and collect results.")
        st.info("💡 **Tip:** Set `GROQ_API_KEY` environment variable to generate real LLM questions. Without it, the app uses mock questions.")
        
        # Show debug log file if it exists
        if log_file.exists():
            with st.expander("📋 Debug Log"):
                with open(log_file, "r") as f:
                    log_content = f.read()
                    st.code(log_content[-2000:], language="text")  # Show last 2000 chars
        
        if st.button("Start Test"):
            st.session_state.state = "prefs"
            st.rerun()

    if st.session_state.state == "prefs":
        st.header("Test Preferences")
        num = st.radio("Number of questions", options=[10, 20, 30], index=0)
        timer_enabled = st.checkbox("Enable per-question timer", value=True)
        if timer_enabled:
            timer_sec = st.number_input("Seconds per question", min_value=10, max_value=600, value=60, step=5)
        else:
            timer_sec = 0

        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Yes, Start"):
                # generate questions
                with st.spinner("Generating questions..."):
                    questions = try_build_llm_and_generate(num)
                    # normalize labels
                    for idx, q in enumerate(questions, start=1):
                        q.label = f"Q{idx}"
                    st.session_state.questions = questions
                    st.session_state.answers = {}
                    st.session_state.current = 0
                    st.session_state.timer_sec = timer_sec
                    st.session_state.batch_id = time.strftime("%Y-%m-%d-%H%M%S", time.gmtime())
                    # persist (best-effort)
                    persist_batch_if_possible(questions, st.session_state.batch_id)
                st.session_state.state = "test"
                st.rerun()
        with cols[1]:
            if st.button("Cancel"):
                st.session_state.state = "landing"
                st.rerun()

    if st.session_state.state == "test":
        questions: List[GeneratedQuestion] = st.session_state.questions
        idx = st.session_state.current
        total = len(questions)
        q = questions[idx]
        
        # Progress and source badge
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### Question {idx+1} of {total}")
        with col2:
            is_mock = "mock" in q.question_id.lower() or q.topic_id == "mock.topic"
            badge = "🔴 Mock" if is_mock else "🟢 LLM"
            st.markdown(f"**{badge}**")
        
        # Display question prominently
        st.markdown(f"**{q.text}**")
        st.markdown("---")
        
        choice = st.radio("Select your answer:", options=q.options, key=f"choice_{idx}")
        
        # Flag question button
        if st.checkbox("🚩 Flag this question (problem/error)", key=f"flag_{idx}"):
            if "flagged_questions" not in st.session_state:
                st.session_state.flagged_questions = set()
            st.session_state.flagged_questions.add(q.question_id)
            st.info(f"Question {q.label} flagged for review")
        elif q.question_id in st.session_state.get("flagged_questions", set()):
            if st.checkbox("✓ Question flagged", value=True, key=f"flag_unflag_{idx}", disabled=True):
                pass

        # Navigation and submission buttons (BEFORE timer so they're always clickable)
        st.markdown("---")
        col_prev, col_skip, col_finish = st.columns([1, 2, 1])
        
        with col_prev:
            if st.button("⬅️ Previous", disabled=(idx == 0)):
                # Save current answer before going back
                if choice:
                    try:
                        letter = ["A", "B", "C", "D"][q.options.index(choice)]
                    except ValueError:
                        letter = ""
                    st.session_state.answers[q.label] = letter or ""
                st.session_state.current = idx - 1
                st.rerun()
        
        with col_skip:
            if st.button("✓ Mark & Next", key=f"next_{idx}"):
                # Save current answer and move to next
                if choice:
                    try:
                        letter = ["A", "B", "C", "D"][q.options.index(choice)]
                    except ValueError:
                        letter = ""
                    st.session_state.answers[q.label] = letter or ""
                else:
                    st.session_state.answers[q.label] = ""
                
                if idx + 1 < total:
                    st.session_state.current = idx + 1
                    st.rerun()
                else:
                    st.session_state.state = "results"
                    st.rerun()
        
        with col_finish:
            if st.button("🏁 Finish Test"):
                answered_count = len(st.session_state.answers)
                unanswered_count = total - answered_count
                
                if unanswered_count > 0:
                    st.session_state.show_finish_warning = True
                    st.rerun()
                else:
                    # Save current answer and finish
                    if choice:
                        try:
                            letter = ["A", "B", "C", "D"][q.options.index(choice)]
                        except ValueError:
                            letter = ""
                        st.session_state.answers[q.label] = letter or ""
                    st.session_state.state = "results"
                    st.rerun()
        
        # Finish warning modal
        if st.session_state.get("show_finish_warning", False):
            st.warning(f"⚠️ **You have {total - len(st.session_state.answers)} unanswered questions.**")
            warn_col1, warn_col2 = st.columns(2)
            with warn_col1:
                if st.button("Go Back & Answer", key="go_back"):
                    st.session_state.show_finish_warning = False
                    st.rerun()
            with warn_col2:
                if st.button("Finish Anyway", key="finish_anyway"):
                    # Save current answer before finishing
                    if choice:
                        try:
                            letter = ["A", "B", "C", "D"][q.options.index(choice)]
                        except ValueError:
                            letter = ""
                        st.session_state.answers[q.label] = letter or ""
                    st.session_state.show_finish_warning = False
                    st.session_state.state = "results"
                    st.rerun()

        # Timer display (after buttons so it doesn't block them)
        if st.session_state.timer_sec and st.session_state.timer_sec > 0:
            placeholder = st.empty()
            remaining = st.session_state.timer_sec
            start = time.time()
            # simple countdown loop — note this blocks but is acceptable for prototype
            while remaining >= 0:
                placeholder.markdown(f"**Time left:** {int(remaining)}s")
                time.sleep(1)
                elapsed = time.time() - start
                remaining = st.session_state.timer_sec - int(elapsed)
                # allow submission by user; check if they've changed choice
                if st.session_state.get(f"submitted_{idx}", False):
                    break

    if st.session_state.state == "results":
        st.header("Results")
        questions: List[GeneratedQuestion] = st.session_state.questions
        answers: Dict[str, str] = st.session_state.answers
        
        # Count attempted vs unanswered
        attempted = sum(1 for q in questions if answers.get(q.label, "").strip())
        unanswered = len(questions) - attempted
        
        # Try LLM grading if available
        graded: List[GradedAnswer] = []
        try:
            cfg = load_config()
            if cfg.groq_api_key:
                llm = AifLlmClient(api_key=cfg.groq_api_key, model=cfg.groq_model)
                graded = llm.grade_answers(questions, answers)
        except Exception:
            graded = grade_locally(questions, answers)
        
        # ===== PERSIST ANSWERS AND MASTERY TO DYNAMODB =====
        try:
            cfg = load_config()
            db = AifDynamoDb(
                region=cfg.aws_region,
                topic_mastery_table=cfg.dynamodb_topic_mastery_table,
                question_history_table=cfg.dynamodb_question_history_table,
                exam_meta_table=cfg.dynamodb_exam_meta_table,
                daily_question_map_table=cfg.dynamodb_daily_question_map_table,
            )
            
            # Build updates for question_history
            updates = []
            mastery_deltas = {}  # topic_id -> list of deltas
            
            for g in graded:
                updates.append({
                    "question_id": g.question_id,
                    "user_answer": g.user_answer,
                    "is_correct": g.is_correct,
                    "short_feedback": g.short_feedback,
                    "detailed_explanation": g.detailed_explanation,
                })
                
                # Accumulate mastery deltas by topic
                if g.topic_id not in mastery_deltas:
                    mastery_deltas[g.topic_id] = []
                mastery_deltas[g.topic_id].append(g.mastery_delta)
            
            # Batch update question records
            if updates:
                logger.info(f"Persisting {len(updates)} answered questions to DynamoDB...")
                db.batch_update_question_answers(cfg.exam_name, updates)
                logger.info(f"Successfully persisted answers for {len(updates)} questions")
            
            # Update topic mastery scores
            for topic_id, deltas in mastery_deltas.items():
                try:
                    # Get current mastery for this topic
                    current_masteries = db.list_topic_mastery(cfg.exam_name)
                    current_score = next(
                        (m.score for m in current_masteries if m.topic_id == topic_id),
                        0.5  # Default to 0.5 (neutral) if not found
                    )
                    
                    # Calculate new score: clamp between 0.0 and 1.0
                    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
                    new_score = max(0.0, min(1.0, current_score + avg_delta))
                    
                    logger.info(f"Updating mastery for {topic_id}: {current_score:.2f} -> {new_score:.2f} (delta: {avg_delta:.2f})")
                    db.update_topic_mastery(cfg.exam_name, topic_id, new_score)
                except Exception as e:
                    logger.warning(f"Failed to update mastery for {topic_id}: {e}")
            
            if mastery_deltas:
                logger.info(f"Updated mastery for {len(mastery_deltas)} topics")
                st.success(f"✅ Answers and mastery scores persisted to DynamoDB ({len(updates)} answers, {len(mastery_deltas)} topics)")
        except Exception as e:
            logger.error(f"Failed to persist answers to DynamoDB: {e}", exc_info=True)
            st.warning(f"⚠️ Could not persist answers to database: {e}")
        # ===== END DYNAMODB PERSISTENCE =====
        
        # Separate results
        correct_list = [g for g in graded if g.is_correct and answers.get(g.label, "").strip()]
        incorrect_list = [g for g in graded if not g.is_correct and answers.get(g.label, "").strip()]
        unanswered_list = [q for q in questions if not answers.get(q.label, "").strip()]
        
        # Display score card
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correct", len(correct_list))
        with col2:
            st.metric("Incorrect", len(incorrect_list))
        with col3:
            st.metric("Not Attempted", len(unanswered_list))
        
        # Show flagged questions count if any
        flagged = st.session_state.get("flagged_questions", set())
        if flagged:
            st.warning(f"⚠️ **{len(flagged)} question(s) flagged for review**")
        
        st.markdown("---")
        
        # Show results by category
        if correct_list:
            st.subheader("✅ Correct")
            for g in correct_list:
                st.markdown(f"**{g.label}**")
                st.write(g.short_feedback)
        
        if incorrect_list:
            st.subheader("❌ Incorrect")
            for g in incorrect_list:
                st.markdown(f"**{g.label}** — {g.short_feedback}")
                st.write(g.detailed_explanation)
        
        if unanswered_list:
            st.subheader("⏭️ Not Attempted")
            for q in unanswered_list:
                st.markdown(f"**{q.label}** — {q.text}")
        
        # Flagged questions feedback section
        flagged_questions = st.session_state.get("flagged_questions", set())
        if flagged_questions:
            st.markdown("---")
            st.subheader("🚩 Flagged Questions — Submit Feedback")
            st.info("Help improve the question bank by describing the issue with flagged questions.")
            
            for q in questions:
                if q.question_id in flagged_questions:
                    with st.expander(f"Question {q.label}: Feedback", expanded=False):
                        st.write(f"**Question:** {q.text}")
                        st.write(f"**Options:** {', '.join(q.options)}")
                        
                        feedback_type = st.selectbox(
                            "What's the issue?",
                            options=["Select...", "incorrect_answer", "unclear_wording", "factually_wrong", "ambiguous", "other"],
                            key=f"feedback_type_{q.question_id}"
                        )
                        
                        feedback_comment = st.text_area(
                            "Describe the issue:",
                            placeholder="e.g., The correct answer is not in the options...",
                            key=f"feedback_comment_{q.question_id}"
                        )
                        
                        if st.button("Submit Feedback", key=f"submit_feedback_{q.question_id}"):
                            if feedback_type != "Select...":
                                try:
                                    cfg = load_config()
                                    db = AifDynamoDb(
                                        region=cfg.aws_region,
                                        topic_mastery_table=cfg.dynamodb_topic_mastery_table,
                                        question_history_table=cfg.dynamodb_question_history_table,
                                        exam_meta_table=cfg.dynamodb_exam_meta_table,
                                        daily_question_map_table=cfg.dynamodb_daily_question_map_table,
                                    )
                                    db.submit_question_feedback(
                                        exam_name=cfg.exam_name,
                                        question_id=q.question_id,
                                        feedback_type=feedback_type,
                                        comment=feedback_comment or "No details provided",
                                        batch_id=st.session_state.batch_id,
                                    )
                                    st.success("✅ Feedback submitted! Thank you for helping improve the questions.")
                                except Exception as e:
                                    st.warning(f"Could not persist feedback: {e}")
                            else:
                                st.warning("Please select an issue type.")
        
        st.markdown("---")
        if st.button("Back to Landing"):
            st.session_state.state = "landing"
            st.rerun()


if __name__ == "__main__":
    main()
