#!/usr/bin/env bash
# Stage 14 — Integration: one-command local setup.
#
# Usage: bash setup.sh
#
# Trains the NLP models, generates the synthetic dataset (if the real one
# hasn't been downloaded), initializes the SQLite database, and seeds it
# with demo data run through the real NLP pipeline. Does NOT train the RL
# agent (that's a separate, longer step -- see README "Training the RL
# agent").
set -e

echo "=== Emergency Message Prioritization System — Setup ==="

if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    # Generate a real random secret key instead of the placeholder.
    python3 -c "
import re, secrets
with open('.env') as f:
    content = f.read()
content = re.sub(r'SECRET_KEY=.*', f'SECRET_KEY={secrets.token_hex(32)}', content)
with open('.env', 'w') as f:
    f.write(content)
"
    echo ".env created with a freshly generated SECRET_KEY."
fi

echo ""
echo "Step 1/5: Downloading spaCy English model (if not already present)..."
python3 -m spacy download en_core_web_sm || echo "  (spaCy download failed or already present — continuing)"

echo ""
echo "Step 2/5: Attempting to download the real disaster-response dataset..."
python3 -m nlp.data_acquisition || echo "  Falling back to synthetic dataset."

if [ ! -f "data/seed/synthetic_messages.csv" ]; then
    echo "  Generating synthetic fallback dataset..."
    python3 -m nlp.synthetic_data
fi

echo ""
echo "Step 3/5: Training NLP models (category classifier + urgency classifier)..."
python3 -m nlp.classifier --data data/seed/synthetic_messages.csv --algo logreg
python3 -m nlp.urgency --data data/seed/synthetic_messages.csv

echo ""
echo "Step 4/5: Initializing database and seeding demo data..."
python3 -m app.models.seed_data

echo ""
echo "Step 5/5: Running rule-based-vs-random evaluation baseline (RL agent skipped until trained)..."
python3 -m evaluation.compare_policies --episodes 30 --skip-rl

echo ""
echo "=== Setup complete ==="
echo "Start the app with:  uvicorn app.main:app --reload --port 8000"
echo "Then open:           http://127.0.0.1:8000"
echo ""
echo "To also train the RL agent (takes several minutes on CPU):"
echo "  python -m rl.train --timesteps 50000"
echo "  python -m evaluation.compare_policies --episodes 30"
