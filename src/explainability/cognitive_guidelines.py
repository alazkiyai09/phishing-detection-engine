from __future__ import annotations


def build_guidelines(label: str, concepts: list[str]) -> list[str]:
    base = [
        "Verify the sender independently before acting.",
        "Do not follow embedded links from suspicious messages.",
    ]
    if label == "phishing":
        base.append(f"Focus review on: {', '.join(concepts[:3])}.")
    return base
