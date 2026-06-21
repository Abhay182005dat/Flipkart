"""
ECIP FastAPI Application
========================
Main application entry-point.  Loads models on startup, exposes REST
endpoints, and manages WebSocket connections for live updates.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ecip.api.routes import router
from ecip.api.state import AppState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and build indices at startup; clean up on shutdown."""
    logger.info("ECIP API starting — loading models and building indices…")
    state = AppState()
    state.load_all()
    app.state.ecip = state
    logger.info("ECIP API ready.")
    yield
    logger.info("ECIP API shutting down.")


app = FastAPI(
    title="ECIP — Event-Driven Congestion Intelligence Platform",
    description=(
        "Forecasts event impact, computes Event Impact Index (EII), "
        "retrieves similar historical incidents, prioritises responses, "
        "optimally allocates manpower and barricades, and enables "
        "scenario planning."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow dashboard from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}


# Serve dashboard static files (index.html, styles.css, app.js)
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

