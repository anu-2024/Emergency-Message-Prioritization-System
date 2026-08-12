"""
Stage 10 — Database design.

SQLAlchemy 2.0-style models implementing the SRS §12 Data Design and §15
audit-trail requirements. SQLite for development (per SRS §18.1); the
engine URL is swappable via DATABASE_URL for Postgres in a real deployment.
"""
import enum
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Text, Enum as SAEnum,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from app.config import settings

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class UserRole(str, enum.Enum):
    admin = "admin"
    responder = "responder"


class MessageStatus(str, enum.Enum):
    received = "received"
    nlp_processed = "nlp_processed"
    prioritized = "prioritized"
    reviewed = "reviewed"
    assigned = "assigned"
    escalated = "escalated"
    resolved = "resolved"
    closed = "closed"


class Urgency(str, enum.Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"
    Critical = "Critical"


class User(Base):
    """Responders and admins. Passwords are always stored hashed
    (see app/services/auth_service.py) -- never plaintext."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SAEnum(UserRole), nullable=False, default=UserRole.responder)
    full_name = Column(String(128), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)

    actions = relationship("ResponderAction", back_populates="user")


class Message(Base):
    """Core entity: one incoming emergency message and its full lifecycle
    (SRS §9.2: Received -> Validated -> NLP processed -> Priority assigned
    -> Human reviewed -> Assigned/acted upon -> Resolved/closed)."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    message_id = Column(String(32), unique=True, nullable=False, index=True)
    raw_text = Column(Text, nullable=False)
    source = Column(String(32), default="web_form")  # web_form / csv_import / simulated_stream
    status = Column(SAEnum(MessageStatus), default=MessageStatus.received, index=True)

    # NLP outputs
    category = Column(String(64), nullable=True)
    category_confidence = Column(Float, nullable=True)
    urgency = Column(SAEnum(Urgency), nullable=True)
    urgency_score = Column(Float, nullable=True)
    urgency_confidence = Column(Float, nullable=True)
    locations = Column(Text, nullable=True)          # JSON-encoded list
    assistance_types = Column(Text, nullable=True)    # JSON-encoded list
    is_duplicate = Column(Boolean, default=False)
    duplicate_of_message_id = Column(String(32), nullable=True)
    duplicate_similarity = Column(Float, nullable=True)

    # Prioritization outputs
    rule_based_priority = Column(Float, nullable=True)
    rl_priority = Column(Float, nullable=True)
    final_priority_source = Column(String(16), nullable=True)  # 'rule_based' | 'rl' | 'human_override'
    human_override_priority = Column(Float, nullable=True)

    received_at = Column(DateTime, default=utcnow, index=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    actions = relationship("ResponderAction", back_populates="message", cascade="all, delete-orphan")


class ResponderAction(Base):
    """Audit trail: every human action taken on a message (SRS §15:
    'Maintain an audit trail for priority overrides and administrative
    actions'). Immutable, append-only -- never updated or deleted."""
    __tablename__ = "responder_actions"

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action_type = Column(String(32), nullable=False)  # review/assign/escalate/resolve/override
    previous_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    note = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utcnow, index=True)

    message = relationship("Message", back_populates="actions")
    user = relationship("User", back_populates="actions")


class EvaluationRun(Base):
    """Stores Stage 9 baseline-vs-RL evaluation snapshots so the dashboard
    can display evaluation metrics historically (SRS: 'View evaluation
    metrics')."""
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True)
    run_at = Column(DateTime, default=utcnow)
    policy_name = Column(String(32), nullable=False)  # random / rule_based / rl_agent
    mean_reward = Column(Float, nullable=False)
    std_reward = Column(Float, nullable=False)
    mean_messages_served = Column(Float, nullable=False)
    mean_avg_served_urgency = Column(Float, nullable=False)
    n_episodes = Column(Integer, nullable=False)


# --- Engine / session setup -------------------------------------------------
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False}
                        if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables. Safe to call repeatedly (no-op if they exist)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a session, always closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
