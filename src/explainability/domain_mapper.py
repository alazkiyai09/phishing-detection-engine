from __future__ import annotations


DOMAIN_MAP = {
    "url_count": "link density",
    "keyword_hits": "social engineering language",
    "has_reply_to_mismatch": "sender inconsistency",
    "sender_looks_free_mail": "infrastructure reputation",
    "uppercase_ratio": "urgency styling",
}


def map_features_to_concepts(features: list[dict[str, float | str]]) -> list[str]:
    return [DOMAIN_MAP.get(str(item["feature"]), str(item["feature"])) for item in features]
