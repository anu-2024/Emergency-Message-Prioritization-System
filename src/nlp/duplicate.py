"""
Duplicate detection (SRS 5.5).

Vectorizes incoming messages (TF-IDF) and computes cosine similarity against
previously seen messages, flagging a message as a likely duplicate when
similarity exceeds a configurable threshold. Surfaces the best-matching
prior message for human verification (the system never auto-discards a
message; a human always reviews the flag).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp.preprocessing import clean_text


@dataclass
class DuplicateMatch:
    is_duplicate: bool
    similarity: float
    best_match_text: str | None
    best_match_id: object | None
    threshold: float


class DuplicateDetector:
    """Stateful, in-memory duplicate detector.

    Call `add(message_id, text)` for every message ingested, and `check(text)`
    before adding a new one to see if it looks like a duplicate of something
    already in the store.
    """

    def __init__(self, threshold: float = 0.82):
        self.threshold = threshold
        self._ids: list = []
        self._raw_texts: list[str] = []
        self._cleaned_texts: list[str] = []

    def add(self, message_id, text: str) -> None:
        self._ids.append(message_id)
        self._raw_texts.append(text)
        self._cleaned_texts.append(clean_text(text))

    def check(self, text: str) -> DuplicateMatch:
        if not self._cleaned_texts:
            return DuplicateMatch(False, 0.0, None, None, self.threshold)

        cleaned = clean_text(text)
        corpus = self._cleaned_texts + [cleaned]
        vectorizer = TfidfVectorizer()
        try:
            tfidf = vectorizer.fit_transform(corpus)
        except ValueError:
            # Empty vocabulary (e.g. all-stopword text) — cannot compare.
            return DuplicateMatch(False, 0.0, None, None, self.threshold)

        sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        is_dup = best_sim >= self.threshold

        return DuplicateMatch(
            is_duplicate=is_dup,
            similarity=best_sim,
            best_match_text=self._raw_texts[best_idx] if is_dup else self._raw_texts[best_idx],
            best_match_id=self._ids[best_idx],
            threshold=self.threshold,
        )

    def __len__(self):
        return len(self._raw_texts)

    def reset(self):
        self._ids.clear()
        self._raw_texts.clear()
        self._cleaned_texts.clear()
