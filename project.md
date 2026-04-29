# AWS AIF Study Agent

## Overview
This project is an autonomous AI study agent designed to help a user prepare for the AWS Artificial Intelligence Foundations (AIF) exam.

The agent sends daily practice questions via email, evaluates user responses, tracks topic mastery over time, and adapts future questions based on performance and time remaining until the exam.

This is a true **stateful AI agent**, not just a scheduled script.

---

## Core Goals
- Maximize probability of passing AWS AIF
- Provide consistent daily practice
- Adapt content to user weaknesses
- Require minimal manual intervention

---

## What Makes This an Agent
- Persistent memory (SQLite/DynamoDB)
- Autonomous daily execution
- Goal-directed behavior
- Feedback-driven adaptation
- LLM used as a tool, not as memory

---

## System Architecture

### Components
- **Scheduler**: Triggers daily agent run
- **SQLite Database**: Long-term memory
- **LLM**: Generates practice questions
- **Email System**: Outbound questions + inbound answers
- **Policy Logic**: Controls topic selection and difficulty
- **Feedback Loop**: Updates mastery scores

---

## Memory Model

### Stored State
- Topic mastery scores
- Question history
- Exam date
- Daily question mappings

### Derived State
- Days until exam
- Weak vs strong topics
- Review vs new question ratio

---

## Database Schema

### topic_mastery
Tracks estimated proficiency per exam topic.

### question_history
Stores all answered questions and outcomes.

### exam_meta
Stores exam-level metadata (exam name, date).

### daily_question_map
Maps emailed question numbers to question IDs for reply parsing.

---

## Daily Agent Loop

1. Load memory from database
2. Compute days remaining
3. Identify weak topics
4. Select topics and difficulty mix
5. Generate questions using LLM
6. Email questions to user
7. Await reply

---

## Feedback Loop

1. Receive reply email
2. Parse structured answers
3. Store results in question history
4. Update topic mastery scores
5. Persist updated state

---

## Email Interaction Model

### Outbound
- Questions labeled as Q1, Q2, etc.
- Instructions included for reply format

### Inbound
- Plain-text parsing
- Ignores quoted replies and signatures
- Deterministic mapping to questions

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
Functional MVP agent with closed feedback loop.
