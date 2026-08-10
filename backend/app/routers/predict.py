from fastapi import APIRouter, HTTPException

from app.predictor import predictor

from app.schemas import (
    HealthResponse,
    ReviewRequest,
    ReviewResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok", model_loaded=predictor.is_ready)


@router.post("/predict", response_model=ReviewResponse, tags=["sentiment"])
def predict(request: ReviewRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    return predictor.predict_one(request.text)
