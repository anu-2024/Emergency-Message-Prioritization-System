"""
Named Entity Recognition for locations, and a separate transparent
keyword-matching layer for requested assistance types (SRS 5.4).

Uses a pretrained statistical NER model (spaCy `en_core_web_sm`, GPE/LOC/FAC
entity labels) when available. If spaCy or its model is unavailable, falls
back to a documented capitalized-phrase heuristic so the module never raises
and the pipeline keeps running end-to-end.

Assistance-type keywords are ALWAYS produced by the disclosed keyword
matcher below (never the learned NER model) so learned vs. rule-based output
is never conflated, per SRS 5.4.
"""
from __future__ import annotations

import re

_nlp_spacy = None
_SPACY_NER_AVAILABLE = False

# A small set of generic words that are capitalized but are not locations,
# used to reduce false positives in the fallback heuristic.
_FALLBACK_STOP_CAPS = {
    "I", "The", "A", "An", "Please", "Send", "Help", "Need", "We", "Someone",
    "Water", "Fire", "Medical", "Emergency", "Food", "Shelter",
}

ASSISTANCE_KEYWORDS = {
    "medical": ["ambulance", "doctor", "medicine", "injured", "bleeding", "unconscious",
                "hospital", "wound", "pain", "cardiac", "medical help"],
    "food": ["food", "ration", "hungry", "meal", "supplies", "grocery"],
    "water": ["water", "drinking water", "thirsty", "tanker", "contaminated"],
    "shelter": ["shelter", "tent", "homeless", "displaced", "roof", "housing"],
    "rescue": ["rescue", "trapped", "stranded", "boat", "evacuate", "save", "stuck"],
}


def _try_load_spacy_ner():
    global _nlp_spacy, _SPACY_NER_AVAILABLE
    if _nlp_spacy is not None or _SPACY_NER_AVAILABLE:
        return
    try:
        import spacy
        try:
            _nlp_spacy = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            _SPACY_NER_AVAILABLE = True
        except OSError:
            _nlp_spacy = None
    except ImportError:
        _nlp_spacy = None


def extract_locations(text: str) -> list[str]:
    """Extract likely place-name mentions.

    Output is ALWAYS indicative only — never treated as verified geolocation
    (explicitly out of scope per SRS 1.2). The caller/UI is responsible for
    labeling this as such.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    _try_load_spacy_ner()
    if _SPACY_NER_AVAILABLE and _nlp_spacy is not None:
        doc = _nlp_spacy(text)
        locs = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
        # de-duplicate, preserve order
        seen = set()
        out = []
        for loc in locs:
            if loc.lower() not in seen:
                seen.add(loc.lower())
                out.append(loc)
        return out

    # Documented fallback heuristic: consecutive-capitalized-word phrases
    # that are not sentence-initial common words.
    candidates = re.findall(r"(?:(?<=\. )|(?<=^))?\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b", text)
    out = []
    seen = set()
    for c in candidates:
        if c in _FALLBACK_STOP_CAPS:
            continue
        if c.lower() in seen:
            continue
        seen.add(c.lower())
        out.append(c)
    return out[:5]


def extract_assistance_keywords(text: str) -> list[str]:
    """Disclosed, transparent keyword-matching layer identifying requested
    assistance type(s). Deliberately simple and rule-based — kept separate
    from the learned NER output (SRS 5.4)."""
    if not isinstance(text, str):
        return []
    lower = text.lower()
    found = []
    for label, keywords in ASSISTANCE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            found.append(label)
    return found


def ner_backend() -> str:
    _try_load_spacy_ner()
    return "spaCy en_core_web_sm (statistical NER)" if _SPACY_NER_AVAILABLE else \
        "fallback (capitalized-phrase heuristic, spaCy model unavailable)"
