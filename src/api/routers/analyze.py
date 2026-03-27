from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.features.pipeline import extract_email_features, score_url_heuristics
from src.models.ensemble import classify_email, classify_url


router = APIRouter()


class URLRequest(BaseModel):
    url: str


class EmailRequest(BaseModel):
    subject: str = ""
    sender: str = ""
    body: str
    headers: dict[str, Any] = Field(default_factory=dict)


class BatchRequest(BaseModel):
    items: list[EmailRequest]


@router.post("/url")
async def analyze_url(payload: URLRequest, request: Request) -> dict[str, Any]:
    heuristics = score_url_heuristics(payload.url)
    result = classify_url(payload.url, heuristics)
    prediction_id = str(uuid4())
    request.app.state.predictions[prediction_id] = result | {"kind": "url"}
    return {"prediction_id": prediction_id, **result}


@router.post("/email")
async def analyze_email(payload: EmailRequest, request: Request) -> dict[str, Any]:
    email_doc = payload.model_dump()
    features = extract_email_features(email_doc)
    result = classify_email(email_doc, features)
    prediction_id = str(uuid4())
    request.app.state.predictions[prediction_id] = result | {"kind": "email", "features": features}
    return {"prediction_id": prediction_id, "features": features, **result}


@router.post("/batch")
async def analyze_batch(payload: BatchRequest, request: Request) -> dict[str, Any]:
    if not payload.items:
        raise HTTPException(status_code=400, detail="Batch is empty")
    results = []
    for item in payload.items:
        email_doc = item.model_dump()
        features = extract_email_features(email_doc)
        result = classify_email(email_doc, features)
        prediction_id = str(uuid4())
        request.app.state.predictions[prediction_id] = result | {"kind": "email", "features": features}
        results.append({"prediction_id": prediction_id, **result})
    return {"count": len(results), "results": results}
