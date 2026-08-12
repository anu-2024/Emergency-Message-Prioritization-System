"""Pydantic request/response schemas for the FastAPI layer."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class MessageCreate(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)
    source: str = "web_form"


class NLPAnalysis(BaseModel):
    category: str
    category_confidence: float
    urgency: str
    urgency_score: float
    urgency_confidence: float
    locations: List[str]
    location_disclaimer: str
    assistance_types: List[str]
    is_duplicate: bool
    duplicate_of: Optional[str] = None
    duplicate_similarity: Optional[float] = None


class MessageOut(BaseModel):
    message_id: str
    raw_text: str
    status: str
    category: Optional[str] = None
    category_confidence: Optional[float] = None
    urgency: Optional[str] = None
    urgency_score: Optional[float] = None
    locations: Optional[List[str]] = None
    assistance_types: Optional[List[str]] = None
    is_duplicate: bool = False
    duplicate_of_message_id: Optional[str] = None
    rule_based_priority: Optional[float] = None
    rl_priority: Optional[float] = None
    final_priority_source: Optional[str] = None
    human_override_priority: Optional[float] = None
    received_at: datetime

    class Config:
        from_attributes = True


class OverrideRequest(BaseModel):
    new_priority: float = Field(..., ge=0.0, le=1.0)
    note: Optional[str] = None


class ActionRequest(BaseModel):
    action_type: str = Field(..., pattern="^(review|assign|escalate|resolve)$")
    note: Optional[str] = None


class QueueStats(BaseModel):
    total_messages: int
    by_status: dict
    by_urgency: dict
    by_category: dict
    duplicate_count: int
