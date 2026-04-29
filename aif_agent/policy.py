from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List

from .config import days_until_exam, AgentConfig
from .db import TopicMastery


# Static domain weightings from weightage.txt (fractions summing to 1.0)
DOMAIN_WEIGHTINGS: Dict[str, float] = {
    "D1": 0.20,
    "D2": 0.24,
    "D3": 0.28,
    "D4": 0.14,
    "D5": 0.14,
}


@dataclass
class TopicSelection:
    topic_id: str
    weight: float


def pick_question_count(cfg: AgentConfig, current_days_left: int) -> int:
    # Slight ramp: closer to exam -> more questions within min/max range
    span = max(cfg.max_questions_per_day - cfg.min_questions_per_day, 0)
    if span == 0:
        return cfg.min_questions_per_day

    # Assume 30-day prep window by default for scaling
    max_window = 30
    factor = 1.0 - min(current_days_left, max_window) / max_window
    return int(cfg.min_questions_per_day + span * factor)


def select_weak_topics(masteries: List[TopicMastery], top_n: int = 10) -> List[str]:
    """Select weaker topics, with a preference for those not asked recently.

    This is a minimal-RCU strategy that uses only the existing
    TopicMastery records. It combines:
    - mastery score (lower = weaker, so higher priority), and
    - recency based on ``last_updated`` (older = higher priority).

    No additional DynamoDB reads are introduced: we work entirely from the
    ``masteries`` list that is already loaded once per run.
    """

    if not masteries:
        return []

    today = date.today()

    def days_since(last_updated: str) -> int:
        if not last_updated:
            # Treat topics without history as "never asked" -> very old
            return 365
        try:
            dt = datetime.fromisoformat(last_updated).date()
        except ValueError:
            # If parsing fails for any reason, do not penalize
            return 365
        delta = (today - dt).days
        return max(delta, 0)

    def recency_penalty(last_updated: str) -> float:
        """Penalty added to the score for very recent topics.

        The higher the penalty, the less likely the topic is to be picked
        again immediately, helping to reduce repetition across days while
        still respecting overall weakness.
        """

        d = days_since(last_updated)
        if d <= 0:
            # Asked today
            return 0.5
        if d == 1:
            # Asked yesterday
            return 0.4
        if 2 <= d <= 3:
            # Asked within the last few days
            return 0.2
        # Older than that: no penalty
        return 0.0

    def priority(m: TopicMastery) -> float:
        return m.score + recency_penalty(m.last_updated)

    sorted_topics = sorted(masteries, key=lambda m: (priority(m), m.score))
    return [m.topic_id for m in sorted_topics[:top_n]]


def domain_weighting() -> Dict[str, float]:
    return DOMAIN_WEIGHTINGS.copy()
