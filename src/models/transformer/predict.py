from src.models.transformer.distilbert import DistilBERTPhishingModel


def predict_text(text: str) -> dict:
    return DistilBERTPhishingModel().predict(text)
