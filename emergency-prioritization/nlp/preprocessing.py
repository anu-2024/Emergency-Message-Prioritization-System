"""
Stage 2 — Text preprocessing.

Cleaning and normalization shared by every downstream NLP component
(classifier, urgency scorer, NER, duplicate detector). Kept dependency-light
(regex + optional spaCy) so it runs fast on CPU.
"""
import re
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z0-9\s.,!?']")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions/extra punctuation, collapse whitespace.

    This is intentionally conservative: it keeps basic punctuation because
    urgency cues ("help!!", "please") and sentence boundaries matter for
    downstream models.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=1)
def _get_spacy_model():
    """Lazily load spaCy's small English model (loaded once, cached)."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.warning(
            "spaCy model 'en_core_web_sm' not found. Run: "
            "python -m spacy download en_core_web_sm"
        )
        raise


def tokenize_lemmatize(text: str, remove_stopwords: bool = True) -> list:
    """Tokenize + lemmatize using spaCy. Falls back to whitespace split
    if the spaCy model isn't installed (keeps the pipeline runnable)."""
    try:
        nlp = _get_spacy_model()
        doc = nlp(text)
        tokens = [
            t.lemma_ for t in doc
            if not t.is_punct and not t.is_space
            and (not remove_stopwords or not t.is_stop)
        ]
        return tokens
    except Exception:  # noqa: BLE001
        logger.debug("Falling back to whitespace tokenization")
        return clean_text(text).split()


def preprocess_for_classifier(text: str) -> str:
    """Full pipeline used before TF-IDF vectorization: clean -> lemmatize -> join."""
    cleaned = clean_text(text)
    tokens = tokenize_lemmatize(cleaned)
    return " ".join(tokens)
