from __future__ import annotations

from typing import Any


def _normalize(score: float) -> float:
    return max(0.0, min(round(score, 4), 1.0))


def classify_url(url: str, heuristics: dict[str, Any]) -> dict[str, Any]:
    raw_score = (
        (0.25 if heuristics.get("has_ip_address") else 0.0)
        + min(float(heuristics.get("subdomain_count", 0)) * 0.08, 0.24)
        + min(float(heuristics.get("special_char_count", 0)) * 0.03, 0.18)
        + min(float(heuristics.get("suspicious_term_count", 0)) * 0.1, 0.3)
    )
    score = _normalize(raw_score)
    return {
        "label": "phishing" if score >= 0.5 else "legitimate",
        "confidence": score,
        "evidence": heuristics,
        "source": "url-heuristics",
        "url": url,
    }


def classify_email(email_doc: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    raw_score = (
        min(float(features.get("keyword_hits", 0)) * 0.11, 0.44)
        + min(float(features.get("url_count", 0)) * 0.12, 0.24)
        + (0.15 if features.get("has_reply_to_mismatch") else 0.0)
        + (0.08 if features.get("sender_looks_free_mail") else 0.0)
        + min(float(features.get("uppercase_ratio", 0)) * 0.5, 0.1)
    )
    score = _normalize(raw_score)
    return {
        "label": "phishing" if score >= 0.5 else "legitimate",
        "confidence": score,
        "summary": f"Detected {features.get('keyword_hits', 0)} suspicious language cues and {features.get('url_count', 0)} embedded URLs.",
        "source": "ensemble-shell",
        "subject": email_doc.get("subject", ""),
    }
