"""
Combines Stages 3-6 into a single call used by the FastAPI service layer
and the RL environment's feature builder.
"""
import logging
from nlp.classifier import CategoryClassifier
from nlp.urgency import UrgencyClassifier
from nlp.ner import extract_entities
from nlp.duplicate_detection import DuplicateIndex

logger = logging.getLogger(__name__)

_category_model = None
_urgency_model = None


def _get_models():
    global _category_model, _urgency_model
    if _category_model is None:
        _category_model = CategoryClassifier.load()
    if _urgency_model is None:
        _urgency_model = UrgencyClassifier.load()
    return _category_model, _urgency_model


def analyze_message(text: str, duplicate_index: DuplicateIndex = None,
                     message_id: str = None) -> dict:
    """Run the full NLP pipeline on one message. Returns a structured
    result matching the SRS §10.3 NLP output schema."""
    category_model, urgency_model = _get_models()

    category_result = category_model.predict_one(text)
    urgency_result = urgency_model.predict_one(text)
    entity_result = extract_entities(text)

    duplicate_result = {"is_duplicate": False, "similarity": 0.0}
    if duplicate_index is not None:
        duplicate_result = duplicate_index.check_duplicate(text)
        if message_id is not None:
            duplicate_index.add(message_id, text)

    return {
        "category": category_result["category"],
        "category_confidence": category_result["confidence"],
        "urgency": urgency_result["urgency"],
        "urgency_score": urgency_result["urgency_score"],
        "urgency_confidence": urgency_result["confidence"],
        "locations": entity_result["locations"],
        "location_disclaimer": entity_result["location_disclaimer"],
        "assistance_types": entity_result["assistance_types"],
        "is_duplicate": duplicate_result["is_duplicate"],
        "duplicate_of": duplicate_result.get("best_match_id"),
        "duplicate_similarity": duplicate_result.get("similarity"),
    }
