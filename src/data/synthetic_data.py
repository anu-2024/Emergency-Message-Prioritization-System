"""
Dataset acquisition for the NLP subsystem (SRS 7.1).

SRS 7.1 requires:
  - support for training on a real, public disaster-response message dataset,
    acquired programmatically, AND
  - a clearly-labeled synthetic fallback dataset so the system remains fully
    runnable without internet access, with every record tagged with its
    provenance ("real" vs "synthetic").

This module tries a best-effort, short-timeout download of a public disaster
message dataset. If that fails for ANY reason (no internet, blocked network,
schema drift, timeout) it silently and deterministically falls back to a
generated synthetic dataset. The synthetic generator is the default and
primary path used throughout this project so the system is guaranteed to run
fully offline.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CATEGORIES, URGENCY_LEVELS, RANDOM_SEED, DATASET_PATH

# ---------------------------------------------------------------------------
# Synthetic message templates
# ---------------------------------------------------------------------------
# Each category has: (subject phrases, location phrases, urgency-tier phrase
# banks). Urgency is derived from a transparent keyword-tier heuristic — this
# is disclosed in the UI per SRS 5.3 (no public dataset provides ground-truth
# urgency labels, so the label source must always be disclosed).

_LOCATIONS = [
    "Koramangala", "Whitefield", "Sector 12", "Riverside Colony", "Old Town",
    "Marina Road", "Green Valley", "North Bridge", "Lakeview Apartments",
    "Central Market", "Hilltop Nagar", "East Ward", "Station Road",
    "Palm Grove", "Sunrise Township", "Harbor District", "Millbrook",
    "Fairview Estate", "Rosewood Lane", "Kings Cross",
]

_CRITICAL_WORDS = ["unconscious", "not breathing", "trapped", "collapsed", "bleeding heavily",
                    "drowning", "on fire", "buried under rubble", "cardiac arrest", "cannot move"]
_HIGH_WORDS = ["severe pain", "rising water fast", "smoke everywhere", "injured", "stranded",
               "roof caving in", "elderly stuck", "child missing", "gas leak"]
_MEDIUM_WORDS = ["needs help soon", "water rising slowly", "minor injury", "low on supplies",
                 "power has been out", "road partially blocked", "feeling unwell"]
_LOW_WORDS = ["just checking in", "small crack in wall", "minor leak", "requesting information",
              "non-urgent request", "when will help arrive", "general inquiry"]

_TEMPLATES = {
    "Medical": [
        "{person} is {crit} near {loc}, please send medical help",
        "Need an ambulance at {loc}, {person} {crit}",
        "Medical emergency at {loc} - {person} {crit}",
        "Someone collapsed at {loc}, {crit}, need doctor urgently",
    ],
    "Flood/Rescue": [
        "Water levels {crit} at {loc}, families are stranded",
        "Flooding at {loc}, {person} {crit}, need boat rescue",
        "We are trapped by flood water near {loc}, {crit}",
        "River overflowing near {loc}, {crit}, send rescue team",
    ],
    "Fire": [
        "Fire reported at {loc}, {crit}, need fire brigade immediately",
        "Building {crit} at {loc}, smoke visible for miles",
        "Fire spreading fast near {loc}, {person} {crit}",
        "Kitchen fire at {loc}, {crit}, please respond",
    ],
    "Food": [
        "Families at {loc} have run out of food, {crit}",
        "Need food supplies delivered to {loc}, {crit}",
        "No food distribution has reached {loc} yet, {crit}",
        "Requesting ration kits for {loc}, {crit}",
    ],
    "Water": [
        "Drinking water contaminated at {loc}, {crit}",
        "No clean water available at {loc} for two days, {crit}",
        "Water supply cut off at {loc}, {crit}, many households affected",
        "Need water tanker sent to {loc}, {crit}",
    ],
    "Shelter": [
        "Homes destroyed near {loc}, families need shelter, {crit}",
        "Temporary shelter overcrowded at {loc}, {crit}",
        "Displaced families waiting outside {loc}, {crit}",
        "Requesting tents and shelter material at {loc}, {crit}",
    ],
    "Infrastructure": [
        "Bridge near {loc} is damaged, {crit}, road unsafe",
        "Power lines down at {loc}, {crit}",
        "Main road to {loc} blocked by debris, {crit}",
        "Communication tower near {loc} is down, {crit}",
    ],
    "Other/Irrelevant": [
        "Just wanted to say thank you to the volunteers at {loc}",
        "Is there a helpline number for general questions about {loc}?",
        "Sharing an update, everything seems calm near {loc} today",
        "Random test message, please ignore, sent from {loc}",
    ],
}

_PERSONS = ["my father", "my neighbor", "an elderly woman", "a child", "my sister",
            "a group of people", "the shopkeeper", "my whole family", "an unknown man"]


def _urgency_phrase(level: str) -> str:
    bank = {
        "Critical": _CRITICAL_WORDS,
        "High": _HIGH_WORDS,
        "Medium": _MEDIUM_WORDS,
        "Low": _LOW_WORDS,
    }[level]
    return random.choice(bank)


def _urgency_distribution_for_category(category: str) -> list[str]:
    """Different categories skew toward different urgency tiers, mirroring
    real disaster-message data (e.g. medical/fire skew critical, 'Other'
    skews low)."""
    if category in ("Medical", "Fire"):
        weights = [0.05, 0.20, 0.35, 0.40]
    elif category in ("Flood/Rescue",):
        weights = [0.08, 0.22, 0.35, 0.35]
    elif category == "Other/Irrelevant":
        weights = [0.70, 0.22, 0.06, 0.02]
    else:
        weights = [0.20, 0.35, 0.30, 0.15]
    return random.choices(URGENCY_LEVELS, weights=weights, k=1)


def generate_synthetic_dataset(n_samples: int = 1600, seed: int = RANDOM_SEED,
                                duplicate_fraction: float = 0.06) -> pd.DataFrame:
    """Generate a labeled synthetic emergency-message dataset.

    Every row is tagged ``provenance="synthetic"`` (SRS 7.1) and urgency rows
    are additionally tagged with the heuristic label source (SRS 5.3).
    """
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    per_category = max(1, n_samples // len(CATEGORIES))
    for category in CATEGORIES:
        templates = _TEMPLATES[category]
        for _ in range(per_category):
            level = _urgency_distribution_for_category(category)[0]
            template = rng.choice(templates)
            text = template.format(
                person=rng.choice(_PERSONS),
                loc=rng.choice(_LOCATIONS),
                crit=_urgency_phrase(level),
            )
            rows.append({
                "text": text,
                "category": category,
                "urgency_level": level,
                "provenance": "synthetic",
                "urgency_label_source": "keyword_tier_heuristic",
            })

    df = pd.DataFrame(rows)

    # Inject near-duplicate messages so the duplicate-detection module (SRS
    # 5.5) has genuine positive examples to demonstrate against.
    n_dupes = int(len(df) * duplicate_fraction)
    dupe_rows = df.sample(n=n_dupes, random_state=seed).copy()
    dupe_rows["text"] = dupe_rows["text"] + " " + rng.choice(
        ["please respond", "still waiting", "urgent update", "same location as before"]
    )
    df = pd.concat([df, dupe_rows], ignore_index=True)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["message_id"] = range(1, len(df) + 1)
    return df[["message_id", "text", "category", "urgency_level",
               "provenance", "urgency_label_source"]]


def try_download_real_dataset(timeout_seconds: int = 6) -> pd.DataFrame | None:
    """Best-effort attempt to fetch a real public disaster-response message
    dataset (SRS 1.4 references the Figure Eight / Appen dataset).

    Returns None on ANY failure (no internet, timeout, schema mismatch) so
    the caller can fall back to the synthetic dataset. This function must
    never raise.
    """
    url = (
        "https://raw.githubusercontent.com/appen/"
        "disaster-response-messages/main/disaster_response_messages_training.csv"
    )
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout_seconds) as resp:
            raw = pd.read_csv(resp)
        if "message" not in raw.columns:
            return None
        raw = raw.rename(columns={"message": "text"})
        raw["provenance"] = "real"
        raw["urgency_label_source"] = "not_available_in_real_dataset"
        if "category" not in raw.columns:
            return None
        return raw
    except Exception:
        return None


def load_dataset(force_synthetic: bool = False, n_samples: int = 1600,
                  cache: bool = True) -> pd.DataFrame:
    """Load the training dataset: real (best-effort) with synthetic fallback.

    If a cached CSV already exists on disk it is reused so repeated runs
    (notebook re-runs, Streamlit reruns) are fast and deterministic.
    """
    if cache and DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    df = None
    if not force_synthetic:
        df = try_download_real_dataset()

    if df is None:
        df = generate_synthetic_dataset(n_samples=n_samples)

    if cache:
        df.to_csv(DATASET_PATH, index=False)
    return df


if __name__ == "__main__":
    data = load_dataset(cache=True)
    print(f"Loaded {len(data)} messages. Provenance counts:")
    print(data["provenance"].value_counts())
    print(f"Saved to {DATASET_PATH}")
