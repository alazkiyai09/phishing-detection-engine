from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentFinding:
    agent: str
    score: float
    rationale: str


def run_agents(subject: str, body: str) -> list[AgentFinding]:
    combined = f"{subject} {body}".lower()
    return [
        AgentFinding("url", 0.7 if "http" in combined else 0.1, "Looks for embedded links."),
        AgentFinding("content", 0.8 if "verify" in combined or "urgent" in combined else 0.2, "Looks for phishing language."),
        AgentFinding("header", 0.6 if "reply-to" in combined else 0.2, "Looks for header anomalies."),
        AgentFinding("visual", 0.4 if "logo" in combined else 0.1, "Placeholder visual suspicion score."),
    ]
