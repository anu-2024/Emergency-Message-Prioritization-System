"""
Stage 11 — FastAPI backend entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000
Then open http://127.0.0.1:8000 for the dashboard, or
http://127.0.0.1:8000/docs for the interactive API docs.
"""
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.database import init_db, SessionLocal
from app.api import auth, messages, evaluation
from app.api.messages import rebuild_duplicate_index

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="NLP + Reinforcement Learning based emergency message "
                 "prioritization system. Human-in-the-loop decision support "
                 "-- never autonomously dispatches emergency services.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(messages.router)
app.include_router(evaluation.router)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
def on_startup():
    logger.info("Initializing database...")
    init_db()
    db = SessionLocal()
    try:
        rebuild_duplicate_index(db)
        logger.info("Duplicate index rebuilt from existing messages.")
    finally:
        db.close()
    logger.info("%s started.", settings.app_name)


@app.get("/")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "app_name": settings.app_name})


@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "app_name": settings.app_name})


@app.get("/health")
def health_check():
    return {"status": "ok", "app": settings.app_name}
