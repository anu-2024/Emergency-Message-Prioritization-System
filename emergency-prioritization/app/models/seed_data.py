"""
Stage 10b — Database initialization & seed/demo data.

Creates the schema and populates it with:
  * Two demo users: one admin, one responder (SRS §4 stakeholder roles)
  * A batch of demo messages loaded from data/seed/synthetic_messages.csv
    and run through the real NLP pipeline (not fake pre-filled values)

Run with: python -m app.models.seed_data
"""
import json
import logging
from pathlib import Path

import pandas as pd

from app.models.database import init_db, SessionLocal, User, Message, UserRole, MessageStatus, Urgency
from app.services.auth_service import hash_password
from nlp.pipeline import analyze_message
from nlp.duplicate_detection import DuplicateIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "seed" / "synthetic_messages.csv"


def seed_users(db) -> None:
    if db.query(User).count() > 0:
        logger.info("Users already exist, skipping user seeding.")
        return

    admin = User(
        username="admin",
        hashed_password=hash_password("Admin@123"),  # CHANGE after first login
        role=UserRole.admin,
        full_name="System Administrator",
    )
    responder = User(
        username="responder1",
        hashed_password=hash_password("Responder@123"),  # CHANGE after first login
        role=UserRole.responder,
        full_name="Demo Responder",
    )
    db.add_all([admin, responder])
    db.commit()
    logger.info("Seeded demo users: admin/Admin@123, responder1/Responder@123 "
                "(CHANGE THESE PASSWORDS before any real deployment)")


def seed_messages(db, n: int = 40) -> None:
    if db.query(Message).count() > 0:
        logger.info("Messages already exist, skipping message seeding.")
        return
    if not SEED_CSV.exists():
        logger.warning("No seed CSV found at %s, skipping message seeding.", SEED_CSV)
        return

    df = pd.read_csv(SEED_CSV).head(n)
    dup_index = DuplicateIndex()
    logger.info("Running %d demo messages through the real NLP pipeline...", len(df))

    for _, row in df.iterrows():
        try:
            result = analyze_message(row["text"], dup_index, row["message_id"])
        except FileNotFoundError:
            logger.error(
                "NLP models not trained yet. Run: python -m nlp.classifier && "
                "python -m nlp.urgency  -- before seeding messages."
            )
            return

        msg = Message(
            message_id=row["message_id"],
            raw_text=row["text"],
            source="seed_demo_data",
            status=MessageStatus.nlp_processed,
            category=result["category"],
            category_confidence=result["category_confidence"],
            urgency=Urgency(result["urgency"]),
            urgency_score=result["urgency_score"],
            urgency_confidence=result["urgency_confidence"],
            locations=json.dumps(result["locations"]),
            assistance_types=json.dumps(result["assistance_types"]),
            is_duplicate=result["is_duplicate"],
            duplicate_of_message_id=result["duplicate_of"],
            duplicate_similarity=result["duplicate_similarity"],
        )
        db.add(msg)
    db.commit()
    logger.info("Seeded %d demo messages (source=seed_demo_data, NLP-analyzed for real).", len(df))


def run():
    init_db()
    db = SessionLocal()
    try:
        seed_users(db)
        seed_messages(db)
    finally:
        db.close()


if __name__ == "__main__":
    run()
