"""
Text preprocessing (SRS 5.1).

Requirement: normalize via lowercasing, URL/mention removal, punctuation
cleanup, and (where available) lemmatization with stopword removal, with a
documented, functioning fallback if the optional lemmatization dependency
(spaCy) is unavailable.
"""
from __future__ import annotations

import re

# Small, dependency-free English stopword list used by the fallback path so
# preprocessing never hard-depends on spaCy/nltk being importable.
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their", "this",
    "that", "these", "those", "and", "or", "but", "if", "then", "so",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "than", "too", "very", "can", "will", "just", "do", "does", "did",
    "has", "have", "had", "having", "please",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"[@#]\w+")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")

_nlp_spacy = None
_SPACY_AVAILABLE = False


def _try_load_spacy():
    """Lazy, one-time attempt to load a spaCy pipeline for lemmatization.

    Kept lazy so importing this module never fails or pays the spaCy load
    cost when only the fallback path is needed.
    """
    global _nlp_spacy, _SPACY_AVAILABLE
    if _nlp_spacy is not None or _SPACY_AVAILABLE:
        return
    try:
        import spacy
        try:
            _nlp_spacy = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            # Model not downloaded — functioning fallback kicks in below.
            _nlp_spacy = None
            return
        _SPACY_AVAILABLE = True
    except ImportError:
        _nlp_spacy = None


def clean_text(text: str, use_lemmatization: bool = True) -> str:
    """Normalize a raw emergency message into a clean token string.

    Pipeline: lowercase -> strip URLs/mentions -> strip punctuation/digits ->
    lemmatize + remove stopwords (spaCy) OR fallback stopword removal only.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    if use_lemmatization:
        _try_load_spacy()

    if use_lemmatization and _SPACY_AVAILABLE and _nlp_spacy is not None:
        doc = _nlp_spacy(text)
        tokens = [t.lemma_ for t in doc if t.lemma_ not in _FALLBACK_STOPWORDS and t.lemma_.strip()]
        return " ".join(tokens)

    # Documented fallback: simple whitespace tokenization + stopword removal,
    # no lemmatization. Functionally complete, just less linguistically rich.
    tokens = [tok for tok in text.split() if tok not in _FALLBACK_STOPWORDS]
    return " ".join(tokens)


def preprocessing_backend() -> str:
    """Report which preprocessing backend is active, for UI disclosure."""
    _try_load_spacy()
    return "spaCy (lemmatization + stopword removal)" if _SPACY_AVAILABLE else \
        "fallback (stopword removal only, spaCy unavailable)"
