from __future__ import annotations


def extract_shap_like_contributions(features: dict[str, float | int | bool]) -> list[dict[str, float | str]]:
    scored = []
    for name, value in features.items():
        magnitude = float(value) if not isinstance(value, bool) else (1.0 if value else 0.0)
        scored.append({"feature": name, "contribution": round(min(magnitude / 10.0, 1.0), 4)})
    return sorted(scored, key=lambda item: item["contribution"], reverse=True)[:5]
