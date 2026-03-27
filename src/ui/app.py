from __future__ import annotations


def build_demo_payload(subject: str, sender: str, body: str) -> dict:
    return {"subject": subject, "sender": sender, "body": body}
