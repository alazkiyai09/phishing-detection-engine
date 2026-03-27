class MultiAgentModel:
    def predict(self, payload: dict) -> dict:
        body = str(payload.get("body", "")).lower()
        score = min(body.count("urgent") * 0.2 + body.count("account") * 0.1, 1.0)
        return {"label": int(score >= 0.5), "confidence": score}
