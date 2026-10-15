from pydantic import BaseModel, Field

class RouteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, examples=["my card has not arrived yet"])
    draft_reply: bool = Field(True, description="Also draft a customer reply (Gemini if GEMINI_API_KEY is set, else template)")

class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)

class Candidate(BaseModel):
    intent: str
    confidence: float

class Reply(BaseModel):
    mode: str
    text: str
    model: str | None = None
    llm_error: str | None = None

class RouteResponse(BaseModel):
    request_id: str
    decision: str
    intent: str
    confidence: float
    top_k: list[Candidate]
    pii_redacted: list[str]
    reply: Reply | None = None
    latency_ms: float
    model_version: str
