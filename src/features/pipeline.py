from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlparse


SUSPICIOUS_KEYWORDS = {
    "urgent",
    "verify",
    "password",
    "reset",
    "suspend",
    "invoice",
    "gift",
    "wire",
    "bank",
}


def score_url_heuristics(url: str) -> dict[str, float | int | bool]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    suspicious_terms = sum(term in url.lower() for term in ("login", "verify", "reset", "secure", "account"))
    return {
        "has_ip_address": host.replace(".", "").isdigit(),
        "subdomain_count": max(host.count(".") - 1, 0),
        "special_char_count": sum(ch in url for ch in "@%&="),
        "suspicious_term_count": suspicious_terms,
        "is_shortened_like": any(domain in host for domain in ("bit.ly", "tinyurl", "t.co")),
        "path_length": len(path),
    }


def extract_email_features(email_doc: Mapping[str, object]) -> dict[str, float | int | bool]:
    subject = str(email_doc.get("subject", ""))
    sender = str(email_doc.get("sender", ""))
    body = str(email_doc.get("body", ""))
    combined = f"{subject} {body}".lower()
    urls = [token for token in body.split() if token.startswith(("http://", "https://"))]
    keyword_hits = sum(keyword in combined for keyword in SUSPICIOUS_KEYWORDS)
    return {
        "url_count": len(urls),
        "keyword_hits": keyword_hits,
        "has_reply_to_mismatch": "reply-to" in str(email_doc.get("headers", {})).lower() and sender.lower() not in str(email_doc.get("headers", {})).lower(),
        "sender_looks_free_mail": any(domain in sender.lower() for domain in ("gmail.com", "outlook.com", "yahoo.com")),
        "body_length": len(body),
        "uppercase_ratio": (sum(ch.isupper() for ch in body) / max(len(body), 1)),
    }
