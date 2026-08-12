"""
Stage 2 — Dataset acquisition.

Downloads the real, publicly available disaster-response message dataset
(originally released by Figure Eight / Appen, widely mirrored for the
Udacity "Disaster Response Pipeline" project) and saves it to data/raw/.

Source used here (verified to exist at time of writing):
  https://github.com/canaveensetia/udacity-disaster-response-pipeline
  (MIT licensed mirror of the Figure Eight disaster-response dataset)

This is REAL, human-labeled data: ~26,000 messages sent during actual
disaster events (floods, earthquakes, storms, etc.), each labeled across
36 categories (related, request, aid_related, medical_help, water, food,
shelter, search_and_rescue, floods, earthquake, ...).

IMPORTANT — what this dataset does NOT contain:
  * It has no "urgency" or "priority" label. Urgency labels used later in
    this project (nlp/urgency.py) are derived with a documented, disclosed
    heuristic (keyword + category based), NOT sourced from Figure Eight.
    This is explicitly disclosed everywhere urgency labels are used.
  * It has no location column suitable for direct use — location is
    extracted separately via NER (nlp/ner.py).

If you have no internet access on this machine, or the download fails,
this script exits with a clear message. Run `python nlp/synthetic_data.py`
instead to generate a clearly-labeled SYNTHETIC dataset so the rest of the
pipeline still runs end-to-end.
"""
import sys
import logging
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCES = {
    "disaster_messages.csv": (
        "https://raw.githubusercontent.com/canaveensetia/"
        "udacity-disaster-response-pipeline/master/data/disaster_messages.csv"
    ),
    "disaster_categories.csv": (
        "https://raw.githubusercontent.com/canaveensetia/"
        "udacity-disaster-response-pipeline/master/data/disaster_categories.csv"
    ),
}


def download_dataset() -> bool:
    """Attempt to download both CSV files. Returns True if both succeed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for filename, url in SOURCES.items():
        dest = RAW_DIR / filename
        if dest.exists():
            logger.info("Already present, skipping: %s", dest)
            continue
        try:
            logger.info("Downloading %s -> %s", url, dest)
            urlretrieve(url, dest)
            logger.info("Saved %s (%d bytes)", dest, dest.stat().st_size)
        except URLError as e:
            logger.error("Failed to download %s: %s", url, e)
            all_ok = False
        except Exception as e:  # noqa: BLE001
            logger.error("Unexpected error downloading %s: %s", url, e)
            all_ok = False
    return all_ok


if __name__ == "__main__":
    ok = download_dataset()
    if not ok:
        logger.warning(
            "Real-dataset download did not fully succeed (no internet access, "
            "or the source moved). This is expected on machines without "
            "internet access. Falling back option:\n"
            "  Run: python nlp/synthetic_data.py\n"
            "This generates a synthetic, clearly-labeled dataset of the same "
            "shape so every later stage still works end-to-end. Swap it out "
            "for the real data any time by re-running this script when you "
            "have internet access."
        )
        sys.exit(1)
    logger.info("Dataset acquisition complete.")
