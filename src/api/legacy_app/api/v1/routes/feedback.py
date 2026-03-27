"""
Feedback endpoints for continuous learning.
"""
from fastapi import APIRouter

from src.api.legacy_app.schemas.requests import FeedbackRequest
from src.api.legacy_app.schemas.responses import FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on model predictions for continuous learning.

    Used to collect false positives/negatives for model retraining.
    """
    # TODO: Implement in Phase 4
    return FeedbackResponse(
        success=True,
        message="Feedback received. Thank you for helping improve our models.",
        feedback_id="fb_123"
    )
