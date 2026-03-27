from __future__ import annotations


class DistilBERTPhishingModel:
    def predict(self, text: str) -> dict[str, float | str]:
        score = min(text.lower().count("verify") * 0.15 + text.lower().count("password") * 0.2, 1.0)
        return {"label": "phishing" if score >= 0.5 else "legitimate", "confidence": round(score, 4)}
