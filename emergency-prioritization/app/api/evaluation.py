"""Stage 11 — Evaluation metrics endpoint (serves Stage 9 results.json to the dashboard)."""
import json
from pathlib import Path

from fastapi import APIRouter, Depends

from app.services.auth_service import get_current_user
from app.models.database import User

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "evaluation" / "results.json"


@router.get("/results")
def get_evaluation_results(current_user: User = Depends(get_current_user)):
    if not RESULTS_PATH.exists():
        return {
            "available": False,
            "message": "No evaluation results yet. Run: python -m evaluation.compare_policies",
        }
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    return {"available": True, "results": results}
