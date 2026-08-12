"""
Stage 6 — Duplicate / near-duplicate detection.

Approach: TF-IDF vectorization + cosine similarity against the current
active queue. This is intentionally simple and explainable (an MCA-viva
requirement per the SRS) rather than a black-box embedding model, while
still being a real, working similarity computation — not a rule/lookup.

A message is flagged as a duplicate of an existing one if cosine
similarity exceeds `SIMILARITY_THRESHOLD`. The closest match and its score
are always returned so a responder can see *why* it was flagged (per
SRS §13.2 "related/duplicate messages" requirement).
"""
import logging
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.preprocessing import preprocess_for_classifier

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.65


@dataclass
class DuplicateIndex:
    """In-memory index of already-seen messages for duplicate checking.

    In production this would be backed by the database (see
    app/services/duplicate_service.py which wraps this class and persists
    results); kept as a plain class here so it's independently testable
    and has no DB dependency.
    """
    message_ids: list = field(default_factory=list)
    raw_texts: list = field(default_factory=list)
    _vectorizer: TfidfVectorizer = field(default=None, repr=False)

    def add(self, message_id: str, text: str) -> None:
        self.message_ids.append(message_id)
        self.raw_texts.append(text)

    def check_duplicate(self, text: str, threshold: float = SIMILARITY_THRESHOLD) -> dict:
        """Compare `text` against everything currently indexed.
        Returns is_duplicate, best match id/text and similarity score."""
        if not self.raw_texts:
            return {"is_duplicate": False, "best_match_id": None,
                    "best_match_text": None, "similarity": 0.0}

        corpus = [preprocess_for_classifier(t) for t in self.raw_texts] + \
                 [preprocess_for_classifier(text)]
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        try:
            tfidf = vectorizer.fit_transform(corpus)
        except ValueError:
            # Happens if corpus is all-empty after cleaning; treat as no match.
            return {"is_duplicate": False, "best_match_id": None,
                    "best_match_text": None, "similarity": 0.0}

        query_vec = tfidf[-1]
        existing_vecs = tfidf[:-1]
        sims = cosine_similarity(query_vec, existing_vecs)[0]
        best_idx = sims.argmax()
        best_score = float(sims[best_idx])

        return {
            "is_duplicate": best_score >= threshold,
            "best_match_id": self.message_ids[best_idx],
            "best_match_text": self.raw_texts[best_idx],
            "similarity": round(best_score, 4),
            "threshold_used": threshold,
        }


def pairwise_duplicate_groups(message_ids: list, texts: list,
                               threshold: float = SIMILARITY_THRESHOLD) -> list:
    """Batch utility (used in evaluation/notebooks): groups a full message
    list into duplicate clusters via pairwise cosine similarity."""
    if len(texts) < 2:
        return []
    cleaned = [preprocess_for_classifier(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(cleaned)
    sim_matrix = cosine_similarity(tfidf)

    n = len(texts)
    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and sim_matrix[i, j] >= threshold:
                group.append(j)
                visited[j] = True
        if len(group) > 1:
            groups.append([message_ids[k] for k in group])
    return groups
