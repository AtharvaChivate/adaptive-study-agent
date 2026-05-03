# AWS AIF Study Agent

## Overview
This project is an autonomous AI study agent designed to help a user prepare for the AWS Artificial Intelligence Foundations (AIF) exam.

The agent generates practice questions, collects answers through a local Streamlit UI, evaluates user responses, tracks topic mastery over time, and adapts future questions based on performance and time remaining until the exam.

This is a true **stateful AI agent**, not just a scheduled script.

---

## Core Goals
- Maximize probability of passing AWS AIF
- Provide consistent daily practice
- Adapt content to user weaknesses
- Require minimal manual intervention

---

## What Makes This an Agent
- Persistent memory (DynamoDB)
- Autonomous daily execution
- Goal-directed behavior
- Feedback-driven adaptation
- LLM used as a tool, not as memory

---

## System Architecture

### Components
- **Scheduler**: Triggers daily agent run
- **Streamlit UI**: Main interaction point for presenting questions, capturing answers, and providing real-time feedback
- **DynamoDB**: Long-term memory for question history, topic mastery, exam metadata, and user progress
- **LLM**: Generates practice questions and grades answers
- **Policy Logic**: Controls topic selection and difficulty
- **Feedback Loop**: Updates question history and mastery scores

---

## Memory Model

### Stored State
- Topic mastery scores
- Question history
- Exam date
- Daily question mappings

### Current Tables
- `aif_topic_mastery`: Per-topic mastery scores by exam
- `aif_question_history`: Generated questions and user answer history
- `aif_exam_meta`: Exam-level metadata
- `aif_daily_question_map`: Batch-to-question mapping for deterministic reply parsing

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
Maps session question numbers to question IDs for tracking and result persistence.

---

## Daily Agent Loop

1. Load memory from DynamoDB
2. Compute days remaining
3. Identify weak topics
4. Select topics and difficulty mix
5. Generate questions using LLM
6. Persist the batch to DynamoDB
7. Present questions in the Streamlit UI
8. Await and record user answers through the UI

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
Functional MVP agent with closed feedback loop and a working Streamlit-based answer persistence path.
