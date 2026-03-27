class TransformerModel:
    def predict(self, text: str) -> dict:
        score = min(text.lower().count("verify") * 0.15 + text.lower().count("password") * 0.2, 1.0)
        return {"label": int(score >= 0.5), "confidence": score}
