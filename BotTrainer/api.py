"""
BotTrainer — FastAPI Backend.
================================
REST API wrapping the NLU pipeline for the React frontend.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from BotTrainer.modules.utils import setup_logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
INTENTS_PATH = PROJECT_ROOT / "data" / "intents.json"
EVAL_DATASET_PATH = PROJECT_ROOT / "data" / "eval_dataset.json"

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_pipeline = None
_evaluator = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from BotTrainer.pipeline import NLUPipeline
        _pipeline = NLUPipeline(intents_path=INTENTS_PATH)
    return _pipeline


def _get_evaluator():
    global _evaluator
    if _evaluator is None:
        from BotTrainer.modules.evaluator import Evaluator
        _evaluator = Evaluator()
    return _evaluator


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BotTrainer API …")
    _get_pipeline()  # warm-up
    yield
    logger.info("Shutting down BotTrainer API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BotTrainer API",
    version="1.0.0",
    description="NLU intent classification & entity extraction API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    user_message: str
    intent: str
    confidence: float
    entities: dict[str, str]
    reasoning: str
    timestamp: str
    model: str
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    intents_loaded: int
    model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health-check endpoint."""
    pipeline = _get_pipeline()
    return HealthResponse(
        status="ok",
        intents_loaded=len(pipeline.classifier.intent_names),
        model=pipeline.llm.primary_model,
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Classify intent and extract entities from a message."""
    try:
        pipeline = _get_pipeline()
        result = pipeline.predict(req.message)
        return PredictResponse(**result)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/intents")
async def list_intents():
    """Return all known intent names."""
    pipeline = _get_pipeline()
    return {"intents": pipeline.classifier.intent_names, "count": len(pipeline.classifier.intent_names)}


@app.get("/api/history")
async def get_history():
    """Return cached predictions (in-memory, resets on restart)."""
    pipeline = _get_pipeline()
    cache = pipeline.classifier._cache
    return {"history": list(cache.values()), "count": len(cache)}
