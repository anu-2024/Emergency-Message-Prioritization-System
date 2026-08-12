"""
Stage 5 — Named Entity Recognition.

Extracts two things from a message:
  1. Location mentions — via spaCy's pretrained NER (GPE/LOC/FAC entity
     types). This is genuinely spaCy's statistical NER model, not a
     lookup table. Per SRS §7 and §13, extracted location is explicitly
     "indicative" and must be verified before operational use — the API
     response carries that caveat forward (see app/services).
  2. Assistance-type keywords — a transparent, disclosed keyword-matching
     layer (medical / food / water / shelter / rescue / evacuation), since
     "type of assistance requested" is a task-specific concept spaCy's
     general NER was never trained on. This is intentionally rule-based
     and labeled as such — it is NOT presented as a learned entity model.

If spaCy / its English model isn't installed, location extraction falls
back to a simple capitalized-word heuristic so the pipeline keeps running,
with `method` in the output disclosing which path was used.
"""
import re
import logging
from nlp.preprocessing import _get_spacy_model, clean_text

logger = logging.getLogger(__name__)

ASSISTANCE_KEYWORDS = {
    "medical": ["medical", "doctor", "ambulance", "injured", "bleeding", "hospital",
                "medicine", "first aid", "unconscious", "pregnant"],
    "food": ["food", "hungry", "ration", "meal", "baby food"],
    "water": ["water", "drinking water", "thirsty", "dehydrat"],
    "shelter": ["shelter", "homeless", "displaced", "roof", "tent", "camp"],
    "rescue": ["rescue", "trapped", "stuck", "stranded", "drowning", "evacuate", "evacuation"],
    "fire_response": ["fire", "smoke", "burning", "explosion"],
    "infrastructure": ["power", "electricity", "bridge", "road", "network", "signal"],
}

CAPITALIZED_WORD_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
GENERIC_WORDS = {"The", "This", "That", "Please", "Need", "Someone", "Two", "Three",
                  "Our", "We", "They", "Requesting", "Several", "Many", "Minor"}


def extract_location_fallback(text: str) -> list:
    """Very simple heuristic used only if spaCy is unavailable: pick
    capitalized multi-word tokens that aren't sentence-initial generic
    words. Clearly weaker than statistical NER — disclosed via `method`."""
    candidates = CAPITALIZED_WORD_RE.findall(text)
    return [c for c in candidates if c not in GENERIC_WORDS]


def extract_locations(text: str) -> dict:
    try:
        nlp = _get_spacy_model()
        doc = nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
        return {"locations": locations, "method": "spacy_ner"}
    except Exception:  # noqa: BLE001
        locations = extract_location_fallback(text)
        return {"locations": locations, "method": "capitalization_heuristic_fallback"}


def extract_assistance_types(text: str) -> list:
    """Disclosed keyword-matching layer for assistance type."""
    t = clean_text(text)
    found = []
    for category, keywords in ASSISTANCE_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            found.append(category)
    return found


def extract_entities(text: str) -> dict:
    """Combined entity extraction used by the API layer."""
    loc_result = extract_locations(text)
    assistance = extract_assistance_types(text)
    return {
        "locations": loc_result["locations"],
        "location_method": loc_result["method"],
        "location_disclaimer": "Extracted location is indicative text-based "
                                "NER output and must be verified before "
                                "operational use.",
        "assistance_types": assistance,
        "assistance_method": "keyword_matching",
    }
