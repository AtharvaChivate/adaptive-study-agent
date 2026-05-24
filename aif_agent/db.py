from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List

import boto3
from boto3.dynamodb.conditions import Key, Attr


@dataclass
class TopicMastery:
    exam_name: str
    topic_id: str  # e.g. "domain1.task1.1" or "service:Amazon S3"
    score: float   # 0.0 - 1.0
    last_updated: str


@dataclass
class QuestionFeedback:
    question_id: str
    feedback_type: str  # "incorrect_answer", "unclear", "factually_wrong", "ambiguous"
    comment: str
    submitted_at: str
    batch_id: str | None = None


@dataclass
class RejectedQuestion:
    exam_name: str
    rejected_id: str  # unique id for this rejection event
    batch_id: str  # batch_id from generation attempt
    attempt: int  # attempt number (1, 2, 3)
    question_id: str  # from LLM response (may be malformed)
    label: str
    rejection_reason: str  # detailed reason why it was rejected
    raw_question: Dict[str, Any]  # full question data for debugging
    created_at: str


class AifDynamoDb:
    def __init__(
        self,
        region: str,
        topic_mastery_table: str,
        question_history_table: str,
        exam_meta_table: str,
        daily_question_map_table: str,
        rejected_questions_table: str,
    ) -> None:
        self._resource = boto3.resource("dynamodb", region_name=region)
        self._topic_mastery_tbl = self._resource.Table(topic_mastery_table)
        self._question_history_tbl = self._resource.Table(question_history_table)
        self._exam_meta_tbl = self._resource.Table(exam_meta_table)
        self._daily_question_map_tbl = self._resource.Table(daily_question_map_table)
        self._rejected_questions_tbl = self._resource.Table(rejected_questions_table)

    # --- Exam meta ---

    def get_exam_meta(self, exam_name: str) -> Dict[str, Any] | None:
        resp = self._exam_meta_tbl.get_item(Key={"exam_name": exam_name})
        return resp.get("Item")

    def put_exam_meta(self, exam_name: str, exam_date: date) -> None:
        self._exam_meta_tbl.put_item(
            Item={
                "exam_name": exam_name,
                "exam_date": exam_date.isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
        )

    # --- Topic mastery ---

    def list_topic_mastery(self, exam_name: str) -> List[TopicMastery]:
        resp = self._topic_mastery_tbl.query(
            KeyConditionExpression=Key("exam_name").eq(exam_name)
        )
        items = resp.get("Items", [])
        return [
            TopicMastery(
                exam_name=i["exam_name"],
                topic_id=i["topic_id"],
                score=float(i.get("score", 0.0)),
                last_updated=i.get("last_updated", ""),
            )
            for i in items
        ]

    def update_topic_mastery(self, exam_name: str, topic_id: str, new_score: float) -> None:
        self._topic_mastery_tbl.put_item(
            Item={
                "exam_name": exam_name,
                "topic_id": topic_id,
                # DynamoDB numeric attributes must be Decimal, not float.
                "score": Decimal(str(new_score)),
                "last_updated": datetime.utcnow().isoformat(),
            }
        )

    # --- Daily question map ---

    def put_daily_question_map(
        self,
        exam_name: str,
        batch_id: str,
        mappings: Dict[str, str],  # "Q1" -> question_id
    ) -> None:
        self._daily_question_map_tbl.put_item(
            Item={
                "exam_name": exam_name,
                "batch_id": batch_id,
                "mappings": mappings,
                "created_at": datetime.utcnow().isoformat(),
            }
        )

    def get_daily_question_map(self, exam_name: str, batch_id: str) -> Dict[str, str] | None:
        resp = self._daily_question_map_tbl.get_item(Key={"exam_name": exam_name, "batch_id": batch_id})
        item = resp.get("Item")
        if not item:
            return None
        return item.get("mappings", {})

    # --- Question history ---

    def put_question_history_batch(self, records: List[Dict[str, Any]]) -> None:
        with self._question_history_tbl.batch_writer() as batch:
            for r in records:
                batch.put_item(Item=r)

    def list_questions_for_batch(self, exam_name: str, batch_id: str) -> List[Dict[str, Any]]:
        """Return all question_history records for a given exam and batch.

        Assumes the question_history table has `exam_name` as the partition key.
        We query by exam_name and filter on batch_id. For the daily volume of
        questions (15-20), this is acceptable. For larger scale, consider
        adding a GSI on batch_id.
        """
        resp = self._question_history_tbl.query(
            KeyConditionExpression=Key("exam_name").eq(exam_name),
            FilterExpression=Attr("batch_id").eq(batch_id),
        )
        return resp.get("Items", [])

    # --- Question feedback ---

    def submit_question_feedback(
        self,
        exam_name: str,
        question_id: str,
        feedback_type: str,
        comment: str,
        batch_id: str | None = None,
    ) -> None:
        """Store feedback on a question (incorrect answer, unclear, factually wrong, etc.)."""
        # Store as a separate record with feedback_type as sort key indicator
        self._question_history_tbl.put_item(
            Item={
                "exam_name": exam_name,
                "question_id": f"feedback#{question_id}#{datetime.utcnow().isoformat()}",
                "batch_id": batch_id or "general",
                "feedback_type": feedback_type,
                "comment": comment,
                "submitted_at": datetime.utcnow().isoformat(),
                "original_question_id": question_id,
            }
        )

    def list_question_feedback(self, exam_name: str) -> List[QuestionFeedback]:
        """List all feedback submitted for questions in this exam."""
        resp = self._question_history_tbl.query(
            KeyConditionExpression=Key("exam_name").eq(exam_name)
        )
        items = resp.get("Items", [])
        feedback_list: List[QuestionFeedback] = []
        for item in items:
            if "feedback#" in str(item.get("question_id", "")):
                feedback_list.append(
                    QuestionFeedback(
                        question_id=item.get("original_question_id", ""),
                        feedback_type=item.get("feedback_type", ""),
                        comment=item.get("comment", ""),
                        submitted_at=item.get("submitted_at", ""),
                        batch_id=item.get("batch_id"),
                    )
                )
        return feedback_list

    def update_question_answer(
        self,
        exam_name: str,
        question_id: str,
        user_answer: str,
        is_correct: bool,
        short_feedback: str,
        detailed_explanation: str,
    ) -> None:
        """Update a question record with user's answer and grading results."""
        self._question_history_tbl.update_item(
            Key={"exam_name": exam_name, "question_id": question_id},
            UpdateExpression="SET #answered = :answered, user_answer = :ua, is_correct = :ic, short_feedback = :sf, detailed_explanation = :de, answered_at = :aa",
            ExpressionAttributeNames={"#answered": "answered"},
            ExpressionAttributeValues={
                ":answered": True,
                ":ua": user_answer,
                ":ic": is_correct,
                ":sf": short_feedback,
                ":de": detailed_explanation,
                ":aa": datetime.utcnow().isoformat(),
            },
        )

    def batch_update_question_answers(
        self, 
        exam_name: str, 
        updates: List[Dict[str, Any]]
    ) -> None:
        """Batch update multiple question answers.
        
        Each item in updates dict should have:
        - question_id
        - user_answer
        - is_correct
        - short_feedback
        - detailed_explanation
        """
        for update in updates:
            self._question_history_tbl.update_item(
                    Key={"exam_name": exam_name, "question_id": update["question_id"]},
                    UpdateExpression="SET #answered = :answered, user_answer = :ua, is_correct = :ic, short_feedback = :sf, detailed_explanation = :de, answered_at = :aa",
                    ExpressionAttributeNames={"#answered": "answered"},
                    ExpressionAttributeValues={
                        ":answered": True,
                        ":ua": update["user_answer"],
                        ":ic": update["is_correct"],
                        ":sf": update["short_feedback"],
                        ":de": update["detailed_explanation"],
                        ":aa": datetime.utcnow().isoformat(),
                    },
            )

    # --- Rejected questions (guardrails audit trail) ---

    def put_rejected_question(
        self,
        exam_name: str,
        batch_id: str,
        attempt: int,
        question_id: str,
        label: str,
        rejection_reason: str,
        raw_question: Dict[str, Any],
    ) -> None:
        """Store a rejected question for audit/debugging.
        
        This creates a permanent record of why the question was rejected,
        useful for analyzing LLM quality, guardrails tuning, and debugging.
        """
        from datetime import datetime, timezone
        import uuid
        
        rejected_id = f"{batch_id}#{attempt}#{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        
        self._rejected_questions_tbl.put_item(
            Item={
                "exam_name": exam_name,
                "rejected_id": rejected_id,
                "batch_id": batch_id,
                "attempt": attempt,
                "question_id": question_id,
                "label": label,
                "rejection_reason": rejection_reason,
                "raw_question": raw_question,
                "created_at": now,
            }
        )

    def put_rejected_questions_batch(
        self,
        exam_name: str,
        batch_id: str,
        attempt: int,
        rejected_issues: List[Dict[str, Any]],  # List of {question_id, label, reason, raw_question}
    ) -> None:
        """Batch store multiple rejected questions."""
        with self._rejected_questions_tbl.batch_writer() as batch:
            for issue in rejected_issues:
                from datetime import datetime, timezone
                import uuid
                
                rejected_id = f"{batch_id}#{attempt}#{uuid.uuid4().hex[:8]}"
                now = datetime.now(timezone.utc).isoformat()
                
                batch.put_item(
                    Item={
                        "exam_name": exam_name,
                        "rejected_id": rejected_id,
                        "batch_id": batch_id,
                        "attempt": attempt,
                        "question_id": issue.get("question_id", "unknown"),
                        "label": issue.get("label", "unknown"),
                        "rejection_reason": issue.get("reason", ""),
                        "raw_question": issue.get("raw_question", {}),
                        "created_at": now,
                    }
                )

    def list_rejected_questions(
        self,
        exam_name: str,
        batch_id: str | None = None,
    ) -> List[RejectedQuestion]:
        """List rejected questions for an exam, optionally filtered by batch_id."""
        if batch_id:
            resp = self._rejected_questions_tbl.query(
                KeyConditionExpression=Key("exam_name").eq(exam_name),
                FilterExpression=Attr("batch_id").eq(batch_id),
            )
        else:
            resp = self._rejected_questions_tbl.query(
                KeyConditionExpression=Key("exam_name").eq(exam_name)
            )
        
        items = resp.get("Items", [])
        return [
            RejectedQuestion(
                exam_name=item.get("exam_name", exam_name),
                rejected_id=item.get("rejected_id", ""),
                batch_id=item.get("batch_id", ""),
                attempt=int(item.get("attempt", 0)),
                question_id=item.get("question_id", ""),
                label=item.get("label", ""),
                rejection_reason=item.get("rejection_reason", ""),
                raw_question=item.get("raw_question", {}),
                created_at=item.get("created_at", ""),
            )
            for item in items
        ]
