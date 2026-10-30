"""FastAPI service in front of the SafeSpace LangGraph agent."""

import logging
import uuid
from typing import Any

import agent_graph
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from routing import RiskLevel, Route

logger = logging.getLogger("safespace")

app = FastAPI(title="SafeSpace AI", version="0.2.0")


# ── Request / response models ────────────────────────────────────────────────

class AskRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="The user's message.")
    thread_id: str | None = Field(
        None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$",
        description="Conversation id. Omit to start a new conversation; reuse it to continue one.")


class AskResponse(BaseModel):
    response: str
    tool_called: str
    route: Route
    mood: str
    risk: RiskLevel
    thread_id: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] = []


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ── Structured errors ────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in err["loc"][1:]), "problem": err["msg"]}
               for err in exc.errors()]
    body = ErrorResponse(error=ErrorDetail(code="invalid_request",
                                           message="The request body is not valid.",
                                           details=details))
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    body = ErrorResponse(error=ErrorDetail(
        code="internal_error",
        message="Something went wrong. If you are in danger, call 112 (EU) or your local emergency number."))
    return JSONResponse(status_code=500, content=body.model_dump())


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse,
          responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def ask(query: AskRequest) -> AskResponse:
    # Plain `def`: FastAPI runs it in a worker thread, so a slow model call
    # doesn't block the event loop for other users.
    thread_id = query.thread_id or uuid.uuid4().hex
    state = agent_graph.run_turn(query.message, thread_id)
    return AskResponse(
        response=str(state["messages"][-1].content),
        tool_called=state.get("tool_called", "None"),
        route=state["route"],
        mood=state.get("mood", "neutral"),
        risk=state.get("risk", "NONE"),
        thread_id=thread_id,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
