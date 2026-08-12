"""
Stage 3 — NLP classification.

Classical, CPU-friendly baseline: TF-IDF features + Logistic Regression
(with Linear SVM as an alternative), per the SRS's explicit instruction to
start with classical baselines before heavier transformer models.

This is a REAL, trained model — not a rule-based system pretending to be
ML. Training data currently comes from data/seed/synthetic_messages.csv
(swap to data/raw/ once the real dataset is downloaded; the CLI below
accepts --data to point at any CSV with `text` and `category` columns).
"""
import argparse
import logging
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

from nlp.preprocessing import preprocess_for_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = ARTIFACT_DIR / "category_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "category_classifier_metrics.json"

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "seed" / "synthetic_messages.csv"


class CategoryClassifier:
    """Thin wrapper bundling the fitted vectorizer + classifier together."""

    def __init__(self, vectorizer=None, model=None):
        self.vectorizer = vectorizer
        self.model = model

    def predict(self, texts: list) -> list:
        cleaned = [preprocess_for_classifier(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        return list(self.model.predict(X))

    def predict_proba(self, texts: list):
        cleaned = [preprocess_for_classifier(t) for t in texts]
        X = self.vectorizer.transform(cleaned)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError("Underlying model has no predict_proba (use --algo logreg)")

    def predict_one(self, text: str) -> dict:
        """Convenience method used by the API layer: returns category + confidence."""
        proba = self.predict_proba([text])[0]
        classes = self.model.classes_
        best_idx = proba.argmax()
        return {
            "category": classes[best_idx],
            "confidence": float(proba[best_idx]),
            "all_scores": {c: float(p) for c, p in zip(classes, proba)},
        }

    def save(self):
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        joblib.dump(self.model, MODEL_PATH)
        logger.info("Saved vectorizer -> %s", VECTORIZER_PATH)
        logger.info("Saved model -> %s", MODEL_PATH)

    @classmethod
    def load(cls):
        if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Classifier artifacts not found. Train first: "
                "python -m nlp.classifier --data data/seed/synthetic_messages.csv"
            )
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        return cls(vectorizer=vectorizer, model=model)


def train(data_path: Path, algo: str = "logreg", test_size: float = 0.2) -> dict:
    logger.info("Loading training data from %s", data_path)
    df = pd.read_csv(data_path)
    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError("Training CSV must have 'text' and 'category' columns")
    df = df.dropna(subset=["text", "category"])

    logger.info("Preprocessing %d messages", len(df))
    df["clean_text"] = df["text"].apply(preprocess_for_classifier)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["category"], test_size=test_size,
        random_state=42, stratify=df["category"],
    )

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if algo == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    elif algo == "svm":
        # LinearSVC has no predict_proba; calibrate it so the API layer
        # can still return confidence scores for the dashboard.
        base = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
        model = CalibratedClassifierCV(base, cv=3)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    logger.info("Training %s classifier...", algo)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info("Test accuracy: %.4f | Macro F1: %.4f", acc, f1_macro)

    clf = CategoryClassifier(vectorizer=vectorizer, model=model)
    clf.save()

    metrics = {
        "algo": algo,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": acc,
        "macro_f1": f1_macro,
        "classification_report": report,
        "data_source": str(data_path),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics -> %s", METRICS_PATH)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the emergency category classifier")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                         help="CSV with 'text' and 'category' columns")
    parser.add_argument("--algo", choices=["logreg", "svm"], default="logreg")
    args = parser.parse_args()
    train(args.data, algo=args.algo)
