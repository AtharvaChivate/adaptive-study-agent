from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict


def build_question_email(subject: str, to_email: str, from_email: str, body_text: str) -> MIMEMultipart:
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.attach(MIMEText(body_text, "plain"))
    return msg


def send_email(
    smtp_host: str,
    smtp_port: int,
    username: str | None,
    password: str | None,
    msg: MIMEMultipart,
) -> None:
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        if username and password:
            server.login(username, password)
        server.send_message(msg)


def format_questions_for_email(questions: list[dict]) -> str:
    lines = []
    lines.append("Here are your AWS AIF practice questions for today.\n")
    lines.append("Reply in the following format (one per line):")
    lines.append("Q1: B")
    lines.append("Q2: C")
    lines.append("Q3: A")
    lines.append("")

    import re

    def _clean_option_text(text: str) -> str:
        # Strip leading label like "A)", "B.", "C -" if the model added it.
        return re.sub(r"^[A-D][\)\.:\-\]]\s*", "", text).strip()

    for q in questions:
        lines.append(f"{q['label']}. {q['text']}")
        options = q["options"]
        letters = ["A", "B", "C", "D"]
        for letter, opt in zip(letters, options):
            clean_opt = _clean_option_text(str(opt))
            lines.append(f"  {letter}. {clean_opt}")
        lines.append("")

    return "\n".join(lines)


def parse_reply_body(body: str) -> Dict[str, str]:
    """Parse plain-text reply like `Q1: B` into a dict label -> answer letter.

    Ignores quoted text and signatures by only looking at lines that start with
    something like Q<number>.
    """
    import re

    mapping: Dict[str, str] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or not line.lower().startswith("q"):
            continue
        m = re.match(r"^(q\d+)\s*[:\-=]\s*([a-dA-D])", line)
        if not m:
            continue
        label = m.group(1).upper()
        answer = m.group(2).upper()
        mapping[label] = answer
    return mapping
