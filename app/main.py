"""FastAPI service: confidence-gated banking-query router + optional LLM reply drafting.
Run:  uvicorn app.main:app --reload     Docs: http://127.0.0.1:8000/docs     UI: http://127.0.0.1:8000/"""
from __future__ import annotations
import logging, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from .classifier import IntentRouter
from .llm import ReplyDrafter
from .privacy import redact
from .schemas import BatchRequest, RouteRequest, RouteResponse

log = logging.getLogger("router"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
STATIC = Path(__file__).parent / "static"
state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    state["router"] = IntentRouter(); state["drafter"] = ReplyDrafter()
    log.info("model=%s gemini_enabled=%s", state["router"].version, state["drafter"].enabled)
    yield
    state.clear()

app = FastAPI(title="SWYNEX Banking Query Router API", version="1.0.0", lifespan=lifespan,
              description="Task 2 - integrates the Task 1 intent model and optional Google Gemini reply drafting.")

def _process(text: str, draft: bool) -> dict:
    t0 = time.perf_counter(); clean, pii = redact(text)
    res = state["router"].route([clean])[0]
    reply = state["drafter"].draft(clean, res) if draft else None
    rid = uuid.uuid4().hex[:12]
    # log metadata only - never the customer text
    log.info("req=%s decision=%s intent=%s conf=%.3f pii=%s reply=%s", rid, res["decision"], res["intent"], res["confidence"], pii, reply and reply["mode"])
    return {"request_id": rid, **res, "pii_redacted": pii, "reply": reply,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "model_version": state["router"].version}

@app.get("/health")
def health():
    return {"status": "ok", "model_version": state["router"].version, "llm_enabled": state["drafter"].enabled,
            "gemini_enabled": state["drafter"].enabled,
            "confidence_threshold": state["router"].threshold}

@app.post("/v1/route", response_model=RouteResponse)
def route(req: RouteRequest):
    if not req.text.strip():
        raise HTTPException(422, "text must not be blank")
    return _process(req.text, req.draft_reply)

@app.post("/v1/route/batch")
def route_batch(req: BatchRequest):
    if any(not t.strip() or len(t) > 1000 for t in req.texts):
        raise HTTPException(422, "each text must be 1-1000 non-blank characters")
    return {"count": len(req.texts), "results": [_process(t, False) for t in req.texts]}

@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")
