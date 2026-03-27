from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.explainability.narrative_generator import generate_explanation


router = APIRouter()


@router.post("/{prediction_id}")
async def explain_prediction(prediction_id: str, request: Request) -> dict[str, str]:
    prediction = request.app.state.predictions.get(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    explanation = generate_explanation(prediction)
    return {"prediction_id": prediction_id, "explanation": explanation}


@router.post("/{id}")
async def explain_prediction_by_id(id: str, request: Request) -> dict[str, str]:
    """Compatibility alias for plan endpoint shape."""
    return await explain_prediction(prediction_id=id, request=request)
