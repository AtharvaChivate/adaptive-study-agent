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


class AifDynamoDb:
    def __init__(
        self,
        region: str,
        topic_mastery_table: str,
        question_history_table: str,
        exam_meta_table: str,
        daily_question_map_table: str,
    ) -> None:
        self._resource = boto3.resource("dynamodb", region_name=region)
        self._topic_mastery_tbl = self._resource.Table(topic_mastery_table)
        self._question_history_tbl = self._resource.Table(question_history_table)
        self._exam_meta_tbl = self._resource.Table(exam_meta_table)
        self._daily_question_map_tbl = self._resource.Table(daily_question_map_table)

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
