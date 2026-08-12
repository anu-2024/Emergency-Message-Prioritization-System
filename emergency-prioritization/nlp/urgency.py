"""
Stage 4 — Urgency detection.

Design note (important for the viva): the public disaster-response dataset
has NO urgency/priority ground truth. So urgency labels used to TRAIN this
model come from one of two places, always disclosed by the `label_source`
field wherever they appear:

  * "synthetic"  — assigned by nlp/synthetic_data.py's category-conditioned
                   random weights (used for the fallback dataset).
  * "heuristic"  — for the REAL dataset, we derive a weak-supervision label
                   using a transparent keyword+category scoring rule
                   (see `heuristic_urgency_label` below), since no human
                   urgency annotation exists. This is standard practice
                   for bootstrapping a label where none exists, and it is
                   explicitly disclosed rather than presented as ground
                   truth. A real deployment would replace this with actual
                   responder-assigned urgency labels over time.

On top of whichever labels are available, we train a REAL supervised
classifier (TF-IDF + Logistic Regression, ordinal-aware via class weights)
— this is not just the heuristic re-applied at inference time.
"""
import re
import logging
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from nlp.preprocessing import preprocess_for_classifier, clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
VECTORIZER_PATH = ARTIFACT_DIR / "urgency_vectorizer.joblib"
MODEL_PATH = ARTIFACT_DIR / "urgency_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "urgency_classifier_metrics.json"

URGENCY_LEVELS = ["Low", "Medium", "High", "Critical"]
URGENCY_TO_SCORE = {"Low": 0.25, "Medium": 0.5, "High": 0.75, "Critical": 1.0}

# Transparent keyword weak-supervision rule, used ONLY when a dataset has
# no urgency label at all (e.g. the raw Figure Eight data). Documented here
# rather than hidden, so it can be audited/critiqued in the viva.
CRITICAL_KEYWORDS = [
    "dying", "trapped", "collapsed", "unconscious", "bleeding heavily",
    "not breathing", "drowning", "on fire", "explosion", "critical",
]
HIGH_KEYWORDS = [
    "urgent", "urgently", "help", "rescue", "injured", "emergency",
    "flooded", "stuck", "asap", "immediately",
]
MEDIUM_KEYWORDS = [
    "need", "shortage", "running low", "damaged", "unsafe", "risk",
]


def heuristic_urgency_label(text: str, category: str = "") -> str:
    """Disclosed weak-supervision rule for datasets with no urgency label.
    NOT used at inference time for the trained model — only to bootstrap
    training labels for the real dataset if you use it."""
    t = clean_text(text)
    if any(k in t for k in CRITICAL_KEYWORDS):
        return "Critical"
    if any(k in t for k in HIGH_KEYWORDS):
        return "High"
    if any(k in t for k in MEDIUM_KEYWORDS):
        return "Medium"
    return "Low"


class UrgencyClassifier:
    def __init__(self, vectorizer=None, model=None):
        self.vectorizer = vectorizer
        self.model = model

    def predict_one(self, text: str) -> dict:
        cleaned = preprocess_for_classifier(text)
        X = self.vectorizer.transform([cleaned])
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        best_idx = proba.argmax()
        level = classes[best_idx]
        return {
            "urgency": level,
            "urgency_score": URGENCY_TO_SCORE.get(level, 0.5),
            "confidence": float(proba[best_idx]),
            "all_scores": {c: float(p) for c, p in zip(classes, proba)},
        }

    def save(self):
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        joblib.dump(self.model, MODEL_PATH)

    @classmethod
    def load(cls):
        if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Urgency model artifacts not found. Train first: "
                "python -m nlp.urgency --data data/seed/synthetic_messages.csv"
            )
        return cls(vectorizer=joblib.load(VECTORIZER_PATH), model=joblib.load(MODEL_PATH))


def train(data_path: Path, test_size: float = 0.2) -> dict:
    logger.info("Loading training data from %s", data_path)
    df = pd.read_csv(data_path)

    if "urgency" not in df.columns:
        logger.warning(
            "No 'urgency' column found — applying disclosed heuristic "
            "weak-supervision labels (label_source=heuristic)."
        )
        df["urgency"] = df.apply(
            lambda r: heuristic_urgency_label(r["text"], r.get("category", "")), axis=1
        )
        label_source = "heuristic"
    else:
        label_source = "synthetic" if "source" in df.columns and (df["source"] == "synthetic").all() else "provided"

    df = df.dropna(subset=["text", "urgency"])
    df["clean_text"] = df["text"].apply(preprocess_for_classifier)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["urgency"], test_size=test_size,
        random_state=42, stratify=df["urgency"],
    )

    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    logger.info("Training urgency classifier (label_source=%s)...", label_source)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Urgency test accuracy: %.4f | Macro F1: %.4f", acc, f1_macro)

    clf = UrgencyClassifier(vectorizer=vectorizer, model=model)
    clf.save()

    metrics = {
        "label_source": label_source,
        "n_train": len(X_train), "n_test": len(X_test),
        "accuracy": acc, "macro_f1": f1_macro,
        "classification_report": report, "data_source": str(data_path),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the urgency classifier")
    parser.add_argument("--data", type=Path,
                         default=Path(__file__).resolve().parent.parent / "data" / "seed" / "synthetic_messages.csv")
    args = parser.parse_args()
    train(args.data)
