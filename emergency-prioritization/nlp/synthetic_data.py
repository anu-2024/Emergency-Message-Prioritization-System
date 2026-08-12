"""
Stage 2 — Synthetic fallback dataset generator.

Generates a SYNTHETIC (template + randomization based) dataset of emergency
messages so the full pipeline (NLP training, RL training, dashboard demo)
can run end-to-end even with no internet access to fetch the real dataset.

This is explicitly NOT real disaster data. It is built from message
templates combined with categories, urgency levels, locations and
assistance types, with randomized phrasing, noise and near-duplicates
injected so downstream components (classifier, urgency scorer, NER,
duplicate detector) have realistic signal to learn from.

Every row is tagged `source=synthetic` so it is never confused with the
real dataset (source=real) produced by data_acquisition.py.
"""
import csv
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"
OUTPUT_FILE = SEED_DIR / "synthetic_messages.csv"

random.seed(42)  # reproducible for the academic report

LOCATIONS = [
    "Koramangala", "Whitefield", "Anekal", "Electronic City", "Yelahanka",
    "Mysore Road", "Hebbal", "Marathahalli", "Jayanagar", "Indiranagar",
    "the riverside colony", "Sector 12", "the old bus stand area",
    "the government school", "the highway bridge area",
]

# category -> (urgency_weighted_choices, message templates)
CATEGORIES = {
    "Medical": {
        "templates": [
            "{n} people including {vuln} are badly injured near {loc}, need medical help urgently.",
            "Someone collapsed near {loc}, breathing heavily, please send an ambulance.",
            "We have a pregnant woman in labor near {loc}, no doctor available.",
            "Minor cuts and bruises from the incident near {loc}, first aid would help.",
            "Elderly person with chest pain at {loc}, needs urgent medical attention.",
        ],
        "urgency_weights": {"Critical": 0.4, "High": 0.35, "Medium": 0.2, "Low": 0.05},
    },
    "Flood/Rescue": {
        "templates": [
            "Our street is flooded and {vuln} need rescue near {loc}.",
            "Water level rising fast near {loc}, families stuck on rooftops.",
            "Flood water entering homes at {loc}, need boats for rescue.",
            "Minor waterlogging near {loc}, roads slow but passable.",
            "Trapped in a car due to flooding near {loc}, water rising.",
        ],
        "urgency_weights": {"Critical": 0.35, "High": 0.35, "Medium": 0.2, "Low": 0.1},
    },
    "Fire": {
        "templates": [
            "Fire has broken out near {loc}, spreading fast, people trapped inside.",
            "Small electrical fire near {loc}, under control but need fire dept.",
            "Building on fire at {loc}, multiple people including {vuln} inside.",
            "Smell of smoke near {loc}, not sure of source yet.",
        ],
        "urgency_weights": {"Critical": 0.45, "High": 0.3, "Medium": 0.2, "Low": 0.05},
    },
    "Food": {
        "templates": [
            "No food available for {vuln} near {loc} for two days now.",
            "Families near {loc} running low on food supplies, need help soon.",
            "Requesting food packets for a shelter camp near {loc}.",
            "Ran out of baby food near {loc}, urgent for infants.",
        ],
        "urgency_weights": {"Critical": 0.15, "High": 0.35, "Medium": 0.35, "Low": 0.15},
    },
    "Water": {
        "templates": [
            "No clean drinking water near {loc}, people falling sick.",
            "Water supply cut off near {loc} since yesterday, need water tankers.",
            "Contaminated water near {loc}, urgent need for clean water and purification.",
            "Water levels low at the community tank near {loc}.",
        ],
        "urgency_weights": {"Critical": 0.2, "High": 0.35, "Medium": 0.3, "Low": 0.15},
    },
    "Shelter": {
        "templates": [
            "Homes damaged near {loc}, {vuln} need emergency shelter tonight.",
            "Families displaced near {loc}, looking for temporary shelter.",
            "Roof collapsed partially near {loc}, need shelter for the night.",
            "Camp near {loc} is over capacity, need additional shelter space.",
        ],
        "urgency_weights": {"Critical": 0.2, "High": 0.35, "Medium": 0.3, "Low": 0.15},
    },
    "Infrastructure": {
        "templates": [
            "Bridge near {loc} looks structurally unsafe after the storm.",
            "Power lines down near {loc}, risk of electrocution.",
            "Road completely blocked near {loc} due to fallen trees.",
            "Mobile network down near {loc}, people unable to call for help.",
        ],
        "urgency_weights": {"Critical": 0.25, "High": 0.35, "Medium": 0.3, "Low": 0.1},
    },
    "Other/Irrelevant": {
        "templates": [
            "Just checking in, everyone in my family near {loc} is safe.",
            "Thank you to the volunteers who helped near {loc} yesterday.",
            "Is the market near {loc} open today?",
            "Sharing updates from local news about {loc}, nothing urgent.",
            "Can someone confirm if schools near {loc} are closed tomorrow?",
        ],
        "urgency_weights": {"Critical": 0.0, "High": 0.05, "Medium": 0.25, "Low": 0.7},
    },
}

VULNERABLE_GROUPS = [
    "elderly people", "young children", "a disabled person", "pregnant women",
    "several families", "an infant", "senior citizens",
]

NUMBERS = ["Two", "Three", "Several", "Five", "A dozen", "Many"]


def pick_urgency(weights: dict) -> str:
    levels = list(weights.keys())
    probs = list(weights.values())
    return random.choices(levels, weights=probs, k=1)[0]


def build_message(category: str) -> dict:
    spec = CATEGORIES[category]
    template = random.choice(spec["templates"])
    text = template.format(
        n=random.choice(NUMBERS),
        vuln=random.choice(VULNERABLE_GROUPS),
        loc=random.choice(LOCATIONS),
    )
    urgency = pick_urgency(spec["urgency_weights"])
    return {"text": text, "category": category, "urgency": urgency}


def generate(n_messages: int = 1200, duplicate_rate: float = 0.08) -> list:
    """Generate n_messages synthetic records, injecting near-duplicates."""
    records = []
    categories = list(CATEGORIES.keys())
    msg_id = 1000

    while len(records) < n_messages:
        category = random.choice(categories)
        rec = build_message(category)
        msg_id += 1
        rec["message_id"] = f"MSG-{msg_id}"
        rec["source"] = "synthetic"
        records.append(rec)

        # Occasionally inject a near-duplicate of the message just created,
        # with minor wording changes, to give the duplicate detector signal.
        if random.random() < duplicate_rate and len(records) < n_messages:
            dup_text = rec["text"].replace("near", "close to", 1)
            msg_id += 1
            records.append({
                "message_id": f"MSG-{msg_id}",
                "text": dup_text,
                "category": rec["category"],
                "urgency": rec["urgency"],
                "source": "synthetic",
            })

    random.shuffle(records)
    return records[:n_messages]


def save_csv(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["message_id", "text", "category", "urgency", "source"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info("Wrote %d synthetic records to %s", len(records), path)


if __name__ == "__main__":
    records = generate(n_messages=1200)
    save_csv(records, OUTPUT_FILE)
