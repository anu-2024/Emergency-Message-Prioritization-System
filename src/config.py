"""
Shared constants and filesystem paths.

Kept in one place so the notebook, the RL training script, the Streamlit app
and any future front-end all agree on category names, urgency levels and
where trained artifacts live on disk.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "messages.csv"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.joblib"
URGENCY_VECTORIZER_PATH = MODELS_DIR / "urgency_vectorizer.joblib"
URGENCY_MODEL_PATH = MODELS_DIR / "urgency_model.joblib"
LABEL_ENCODERS_PATH = MODELS_DIR / "label_encoders.joblib"

DQN_MODEL_PATH = MODELS_DIR / "dqn_policy.zip"
EVAL_RESULTS_PATH = MODELS_DIR / "eval_results.json"
TRAINING_CURVE_PATH = MODELS_DIR / "training_curve.json"

# ---------------------------------------------------------------------------
# Domain constants (SRS Section 2.2, 5.2, 5.3)
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Medical",
    "Flood/Rescue",
    "Fire",
    "Food",
    "Water",
    "Shelter",
    "Infrastructure",
    "Other/Irrelevant",
]

URGENCY_LEVELS = ["Low", "Medium", "High", "Critical"]

# Numeric urgency score assigned to each level, mapped into [0, 1] per SRS 5.3
URGENCY_SCORE_MAP = {
    "Low": 0.15,
    "Medium": 0.45,
    "High": 0.75,
    "Critical": 0.95,
}

# Category criticality weight used by the transparent rule-based baseline
# (SRS 6.7). Purely a documented, explainable heuristic — disclosed to the
# user in the UI, never presented as learned/ground truth.
CATEGORY_CRITICALITY = {
    "Medical": 1.00,
    "Fire": 0.95,
    "Flood/Rescue": 0.90,
    "Shelter": 0.60,
    "Water": 0.55,
    "Food": 0.50,
    "Infrastructure": 0.40,
    "Other/Irrelevant": 0.10,
}

RANDOM_SEED = 42
