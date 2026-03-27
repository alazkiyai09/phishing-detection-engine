from src.models.multi_agent.agents import run_agents


def coordinate(subject: str, body: str) -> dict:
    findings = run_agents(subject, body)
    confidence = round(sum(item.score for item in findings) / len(findings), 4)
    return {
        "confidence": confidence,
        "label": "phishing" if confidence >= 0.5 else "legitimate",
        "findings": [item.__dict__ for item in findings],
    }
