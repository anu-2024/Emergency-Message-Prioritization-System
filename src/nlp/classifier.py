"""
Supervised text classifiers (SRS 5.2 Category Classification, 5.3 Urgency
Detection).

Both classifiers use TF-IDF vectorization (unigrams + bigrams) feeding a
Logistic Regression model — classical, CPU-friendly, explainable techniques
as required by SRS 2.4 (Constraints) and 5 (NLP subsystem must genuinely
learn from data, not a relabeled keyword system).

Trained artifacts are persisted with joblib and reloaded at inference time
without retraining (SRS 5.2).
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.nlp.preprocessing import clean_text
from src.config import (
    VECTORIZER_PATH, CATEGORY_MODEL_PATH,
    URGENCY_VECTORIZER_PATH, URGENCY_MODEL_PATH,
    URGENCY_SCORE_MAP, RANDOM_SEED,
)


class _BaseTfidfClassifier:
    """Shared TF-IDF + Logistic Regression training/inference logic."""

    label_col: str
    vectorizer_path = None
    model_path = None

    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.model: LogisticRegression | None = None
        self.classes_: list[str] | None = None

    def fit(self, texts, labels, max_features=6000, C=4.0, test_size=0.2,
            random_state=RANDOM_SEED, verbose=True):
        cleaned = [clean_text(t) for t in texts]
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(cleaned)
        y = np.asarray(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.model = LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced", random_state=random_state
        )
        self.model.fit(X_train, y_train)
        self.classes_ = list(self.model.classes_)

        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        }
        if verbose:
            print(f"[{self.__class__.__name__}] accuracy={metrics['accuracy']:.3f} "
                  f"macro_f1={metrics['macro_f1']:.3f}")
            print(report)
        return metrics, report

    def predict(self, text: str):
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded/trained. Call load() or fit() first.")
        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        proba_map = {cls: float(p) for cls, p in zip(self.model.classes_, proba)}
        confidence = float(max(proba))
        return pred, confidence, proba_map

    def save(self):
        joblib.dump(self.vectorizer, self.vectorizer_path)
        joblib.dump(self.model, self.model_path)

    def load(self):
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)
        self.classes_ = list(self.model.classes_)
        return self

    @classmethod
    def artifacts_exist(cls) -> bool:
        return cls.vectorizer_path.exists() and cls.model_path.exists()


class CategoryClassifier(_BaseTfidfClassifier):
    """Predicts one of the 8 message categories (SRS 5.2)."""
    label_col = "category"
    vectorizer_path = VECTORIZER_PATH
    model_path = CATEGORY_MODEL_PATH


class UrgencyClassifier(_BaseTfidfClassifier):
    """Predicts an urgency level (Low/Medium/High/Critical) and maps it to a
    numeric score in [0, 1] (SRS 5.3).

    The label source is always the transparent keyword-tier heuristic used
    during synthetic dataset generation — this is disclosed to the caller via
    `label_source` since no public dataset provides ground-truth urgency.
    """
    label_col = "urgency_level"
    vectorizer_path = URGENCY_VECTORIZER_PATH
    model_path = URGENCY_MODEL_PATH
    label_source = "keyword_tier_heuristic (disclosed, not human-verified ground truth)"

    def predict_with_score(self, text: str):
        level, confidence, proba_map = self.predict(text)
        score = URGENCY_SCORE_MAP.get(level, 0.5)
        return level, score, confidence, proba_map
