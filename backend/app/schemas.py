from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        examples=["The fabric is soft but the zipper broke on the first day."],
    )


class ReviewResponse(BaseModel):
    review: str
    sentiment: str
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
