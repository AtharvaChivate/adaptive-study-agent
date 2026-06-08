# Autonomous Adaptive Study Agent

## Overview
This project is an autonomous LLM-powered study agent designed to help users master complex technical domains and prepare for specialized certifications.

The agent generates dynamic practice questions, captures answers through an interactive Streamlit UI, evaluates response quality using LLM-driven grading, and utilizes persistent memory to adapt future curricula based on mastery scores and goal deadlines.

This is a true **stateful AI agent**, not just a scheduled script.

---

## Core Goals
- Maximize exam readiness and long-term knowledge retention
- Provide a low-friction, interactive study experience
- Automatically adapt curricula to target individual user weaknesses
- Require minimal manual intervention

---

## What Makes This an Agent
- Persistent memory (DynamoDB)
- Goal-directed behavior
- Feedback-driven adaptation
- **Reliability Layer**: Self-healing loops for LLM API failures
- Deterministic policy engine for curriculum scheduling

---

## System Architecture

### Components
- **Streamlit UI**: Main interaction point for presenting questions, capturing answers, and providing real-time feedback
- **DynamoDB**: Long-term memory for question history, topic mastery, exam metadata, and user progress
- **LLM**: Generates practice questions and grades answers
- **Policy Logic**: Controls topic selection and difficulty
- **Feedback Loop**: Updates question history and mastery scores
- **Guardrails Engine**: Deterministic quality gate for all generated content

---

## Memory Model

### Stored State
- Topic mastery scores
- Question history
- Exam date
- Daily question mappings
- Guardrail rejection logs

### Current Tables
- `aif_topic_mastery`: Per-topic mastery scores by exam
- `aif_question_history`: Audit log of generated questions and user performance
- `aif_exam_meta`: Exam-level metadata
- `aif_daily_question_map`: Mapping for tracking question batches across sessions
- `aif_rejected_questions`: Log for questions that failed quality guardrails

### Derived State
- Days until exam
- Weak vs strong topics
- Review vs new question ratio

---

## Database Schema

### topic_mastery
Tracks estimated proficiency per exam topic.

### question_history
Stores generated questions, user answers, grading results, and feedback.

### exam_meta
Stores exam-level metadata (exam name, date).

### daily_question_map
Maps session labels to unique question IDs for deterministic tracking and persistence.

### rejected_questions
Audit log for questions that failed the guardrails quality gate.

---

## Daily Agent Loop

1. **Retrieve State**: Load mastery scores and goal metadata from DynamoDB.
2. **Analyze Gaps**: Identify weak topics and calculate days remaining until the goal date.
3. **Apply Policy**: The engine selects topics and scales question volume based on progress.
4. **Generate**: Produce questions via LLM with structured output validation.
5. **Validate**: Run content through a deterministic guardrails engine.
6. **Persist**: Write the valid question batch to the data layer.
7. **Interactive Session**: Present questions and capture responses via the Streamlit UI.
8. **Evaluate**: Grade responses and update persistent memory to close the feedback loop.

---

## Feedback Loop

1. Receive user submission through the Streamlit UI
2. Parse and validate answers
3. Store results in question history
4. Update topic mastery scores
5. Persist updated state

---

## UI Interaction Model

### Question Presentation
- Questions displayed with clear labels (Q1, Q2, etc.)
- Topics and difficulty level shown for context
- Instructions provided within the UI

### Answer Capture
- Direct form-based input for user answers
- Real-time validation and feedback
- Immediate storage to DynamoDB upon submission

---

## Design Principles

- Simple over clever
- Deterministic over probabilistic where possible
- LLMs generate content, not policy
- State lives outside the model
- Minimal schema, extensible later

---

## Future Enhancements
- Mistake pattern clustering
- Spaced repetition decay
- Confidence calibration
- Vector search for similar questions
- Web UI dashboard
- Multi-user support

---

## Non-Goals
- Full LMS replacement
- Real-time chat tutoring
- Over-engineered agent frameworks

---

## Status
Functional MVP agent with closed feedback loop, Streamlit UI, and a robust reliability layer for handling LLM output truncation and schema validation retries.
