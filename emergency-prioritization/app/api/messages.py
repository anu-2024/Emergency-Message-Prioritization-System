"""
Stage 11 — Message ingestion, NLP analysis, and priority queue endpoints.
"""
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.database import get_db, Message, MessageStatus, Urgency, ResponderAction, User
from app.models.schemas import MessageCreate, MessageOut, OverrideRequest, ActionRequest, QueueStats
from app.services.auth_service import get_current_user
from app.services.priority_service import compute_final_priority
from nlp.pipeline import analyze_message
from nlp.duplicate_detection import DuplicateIndex

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/messages", tags=["messages"])

# Process-lifetime duplicate index (rebuilt from DB on cold start via
# _rebuild_duplicate_index below, called at app startup in app/main.py).
_dup_index = DuplicateIndex()


def rebuild_duplicate_index(db: Session) -> None:
    _dup_index.message_ids.clear()
    _dup_index.raw_texts.clear()
    for m in db.query(Message).all():
        _dup_index.add(m.message_id, m.raw_text)


def _message_to_out(m: Message) -> MessageOut:
    return MessageOut(
        message_id=m.message_id, raw_text=m.raw_text, status=m.status.value,
        category=m.category, category_confidence=m.category_confidence,
        urgency=m.urgency.value if m.urgency else None, urgency_score=m.urgency_score,
        locations=json.loads(m.locations) if m.locations else [],
        assistance_types=json.loads(m.assistance_types) if m.assistance_types else [],
        is_duplicate=m.is_duplicate, duplicate_of_message_id=m.duplicate_of_message_id,
        rule_based_priority=m.rule_based_priority, rl_priority=m.rl_priority,
        final_priority_source=m.final_priority_source,
        human_override_priority=m.human_override_priority,
        received_at=m.received_at,
    )


@router.post("/submit", response_model=MessageOut)
def submit_message(payload: MessageCreate, db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    """Ingest a new emergency message and run it through the full NLP
    pipeline (classification, urgency, NER, duplicate detection)."""
    message_id = f"MSG-{uuid.uuid4().hex[:10].upper()}"

    try:
        nlp_result = analyze_message(payload.text, _dup_index, message_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    msg = Message(
        message_id=message_id, raw_text=payload.text, source=payload.source,
        status=MessageStatus.nlp_processed,
        category=nlp_result["category"], category_confidence=nlp_result["category_confidence"],
        urgency=Urgency(nlp_result["urgency"]), urgency_score=nlp_result["urgency_score"],
        urgency_confidence=nlp_result["urgency_confidence"],
        locations=json.dumps(nlp_result["locations"]),
        assistance_types=json.dumps(nlp_result["assistance_types"]),
        is_duplicate=nlp_result["is_duplicate"],
        duplicate_of_message_id=nlp_result["duplicate_of"],
        duplicate_similarity=nlp_result["duplicate_similarity"],
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)

    # Compute priority (rule-based always; RL if available) against current queue.
    queue_context = db.query(Message).filter(
        Message.status.in_([MessageStatus.nlp_processed, MessageStatus.prioritized])
    ).all()
    priority = compute_final_priority(msg, queue_context)
    msg.rule_based_priority = priority["rule_based_priority"]
    msg.rl_priority = priority["rl_priority"]
    msg.final_priority_source = priority["final_priority_source"]
    msg.status = MessageStatus.prioritized
    db.commit()
    db.refresh(msg)

    return _message_to_out(msg)


@router.get("/queue", response_model=list[MessageOut])
def get_priority_queue(db: Session = Depends(get_db),
                        current_user: User = Depends(get_current_user)):
    """Returns the ranked queue: human override > RL score > rule-based
    score, descending. This is a SUGGESTED ranking for a human responder --
    never auto-dispatched (SRS §3.2 out-of-scope: autonomous dispatch)."""
    messages = db.query(Message).filter(
        Message.status.notin_([MessageStatus.resolved, MessageStatus.closed])
    ).all()

    def effective_score(m: Message) -> float:
        if m.human_override_priority is not None:
            return m.human_override_priority
        if m.rl_priority is not None:
            return m.rl_priority
        return m.rule_based_priority or 0.0

    messages.sort(key=effective_score, reverse=True)
    return [_message_to_out(m) for m in messages]


@router.get("/{message_id}", response_model=MessageOut)
def get_message(message_id: str, db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    m = db.query(Message).filter(Message.message_id == message_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Message not found")
    return _message_to_out(m)


@router.post("/{message_id}/override", response_model=MessageOut)
def override_priority(message_id: str, payload: OverrideRequest, db: Session = Depends(get_db),
                       current_user: User = Depends(get_current_user)):
    """Human-in-the-loop override -- always available regardless of role,
    always audit-logged (SRS §15 audit trail requirement)."""
    m = db.query(Message).filter(Message.message_id == message_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Message not found")

    previous = m.human_override_priority
    m.human_override_priority = payload.new_priority
    m.final_priority_source = "human_override"

    action = ResponderAction(
        message_id=m.id, user_id=current_user.id, action_type="override",
        previous_value=str(previous), new_value=str(payload.new_priority),
        note=payload.note,
    )
    db.add(action)
    db.commit()
    db.refresh(m)
    return _message_to_out(m)


@router.post("/{message_id}/action", response_model=MessageOut)
def record_action(message_id: str, payload: ActionRequest, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    """Records review/assign/escalate/resolve actions, updates status,
    and appends to the immutable audit trail."""
    m = db.query(Message).filter(Message.message_id == message_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Message not found")

    status_map = {
        "review": MessageStatus.reviewed, "assign": MessageStatus.assigned,
        "escalate": MessageStatus.escalated, "resolve": MessageStatus.resolved,
    }
    previous_status = m.status
    m.status = status_map[payload.action_type]

    action = ResponderAction(
        message_id=m.id, user_id=current_user.id, action_type=payload.action_type,
        previous_value=previous_status.value, new_value=m.status.value, note=payload.note,
    )
    db.add(action)
    db.commit()
    db.refresh(m)
    return _message_to_out(m)


@router.get("/stats/summary", response_model=QueueStats)
def get_stats(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    messages = db.query(Message).all()
    by_status, by_urgency, by_category = {}, {}, {}
    duplicate_count = 0
    for m in messages:
        by_status[m.status.value] = by_status.get(m.status.value, 0) + 1
        if m.urgency:
            by_urgency[m.urgency.value] = by_urgency.get(m.urgency.value, 0) + 1
        if m.category:
            by_category[m.category] = by_category.get(m.category, 0) + 1
        if m.is_duplicate:
            duplicate_count += 1

    return QueueStats(
        total_messages=len(messages), by_status=by_status,
        by_urgency=by_urgency, by_category=by_category,
        duplicate_count=duplicate_count,
    )
