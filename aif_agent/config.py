import os
from datetime import date, datetime
from dataclasses import dataclass


@dataclass
class AgentConfig:
    exam_name: str
    exam_date: date
    min_questions_per_day: int
    max_questions_per_day: int
    days_to_exam_override: int | None
    user_email: str
    from_email: str
    smtp_host: str
    smtp_port: int
    smtp_username: str | None
    smtp_password: str | None
    aws_region: str
    dynamodb_topic_mastery_table: str
    dynamodb_question_history_table: str
    dynamodb_exam_meta_table: str
    dynamodb_daily_question_map_table: str
    groq_api_key: str
    groq_model: str


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_config() -> AgentConfig:
    exam_date_env = os.getenv("AIF_EXAM_DATE")
    days_override = os.getenv("AIF_DAYS_TO_EXAM")

    if not exam_date_env and not days_override:
        raise RuntimeError("Set either AIF_EXAM_DATE (YYYY-MM-DD) or AIF_DAYS_TO_EXAM.")

    exam_date = _parse_date(exam_date_env) if exam_date_env else date.today()

    return AgentConfig(
        exam_name=os.getenv("AIF_EXAM_NAME", "AWS AIF"),
        exam_date=exam_date,
        min_questions_per_day=int(os.getenv("AIF_MIN_QUESTIONS", "15")),
        max_questions_per_day=int(os.getenv("AIF_MAX_QUESTIONS", "20")),
        days_to_exam_override=int(days_override) if days_override else None,
        user_email=os.getenv("AIF_USER_EMAIL", ""),
        from_email=os.getenv("AIF_FROM_EMAIL", os.getenv("AIF_USER_EMAIL", "")),
        smtp_host=os.getenv("AIF_SMTP_HOST", "smtp.gmail.com"),
        smtp_port=int(os.getenv("AIF_SMTP_PORT", "587")),
        smtp_username=os.getenv("AIF_SMTP_USERNAME"),
        smtp_password=os.getenv("AIF_SMTP_PASSWORD"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        dynamodb_topic_mastery_table=os.getenv("AIF_TOPIC_MASTERY_TABLE", "aif_topic_mastery"),
        dynamodb_question_history_table=os.getenv("AIF_QUESTION_HISTORY_TABLE", "aif_question_history"),
        dynamodb_exam_meta_table=os.getenv("AIF_EXAM_META_TABLE", "aif_exam_meta"),
        dynamodb_daily_question_map_table=os.getenv("AIF_DAILY_QUESTION_MAP_TABLE", "aif_daily_question_map"),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
    )


def days_until_exam(cfg: AgentConfig) -> int:
    if cfg.days_to_exam_override is not None:
        return cfg.days_to_exam_override
    today = date.today()
    return max((cfg.exam_date - today).days, 0)
