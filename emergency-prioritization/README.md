# Emergency Message Prioritization System (NLP + Reinforcement Learning)

An MCA academic project: a human-in-the-loop decision-support system that
classifies, scores, and ranks incoming emergency messages using classical
NLP (TF-IDF + Logistic Regression), NER, duplicate detection, and a
Reinforcement Learning agent (DQN via Stable-Baselines3), compared against
a transparent rule-based baseline. Runs entirely on CPU, no GPU required.

**⚠️ This system never autonomously dispatches emergency services.** Every
priority score is a suggestion; a human responder reviews, can override,
and takes the final action. See `docs/ethical_considerations.md`.

---

## What was verified vs. not (read this first)

This project was built in a sandboxed environment **with no internet
access**, so some packages (`gymnasium`, `stable-baselines3`, `torch`,
`fastapi`, `sqlalchemy`, `spacy`, `streamlit`, `pytest`) could not be
installed there. Everything possible was still genuinely executed and
tested in that sandbox:

| Verified by actually running it in the build sandbox | Written correctly but NOT executed there (no internet to install deps) |
|---|---|
| Synthetic dataset generation (1200 records) | FastAPI backend (`app/`) — syntax-checked only |
| Text preprocessing | SQLAlchemy database models — syntax-checked only |
| Category classifier — trained, 100% test acc (synthetic data) | RL training script (`rl/train.py`, SB3 DQN) |
| Urgency classifier — trained, 43% test acc (honest, harder task) | RL inference bridge (`app/services/rl_inference.py`) |
| NER + duplicate detection | Streamlit app — syntax-checked only |
| RL environment (`rl/environment.py`) — via a hand-written Gymnasium API shim | |
| Rule-based baseline — tested | |
| Evaluation harness — ran rule-based (116.8 mean reward) vs random (60.7) | |
| 24 unit tests (`tests/test_nlp.py`, `tests/test_rl.py`) — all pass | API tests (`tests/test_api.py`) — syntax-checked only |

**Run `setup.sh` locally first**, and check its output at each step, before
your project demo/viva. If anything in the untested column has a bug, it
will most likely be a small import-path or API-signature mismatch — the
architecture and logic have been designed and cross-checked carefully, but
"designed carefully" is not the same claim as "ran successfully."

---

## Quick start

```bash
git clone <your-repo-url>
cd emergency-prioritization
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install CPU-only PyTorch first (required by stable-baselines3)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

bash setup.sh                    # trains NLP models, sets up DB, seeds demo data
uvicorn app.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000** — login with `admin` / `Admin@123` or
`responder1` / `Responder@123` (change these before any real deployment).

### Training the RL agent (optional, several minutes on CPU)

```bash
python -m rl.train --timesteps 50000
python -m evaluation.compare_policies --episodes 30
```

This updates `evaluation/results.md` with the RL agent's numbers next to
the rule-based and random baselines.

### Running tests

```bash
pip install pytest
pytest tests/ -v
```

### Running the Streamlit demo

```bash
streamlit run streamlit_app/app.py
```

To deploy on **Streamlit Community Cloud**: push this repo to GitHub,
point the app at `streamlit_app/app.py`. See the comment at the top of
that file for two ways to make sure trained NLP models are available on
first boot (the app auto-trains on the synthetic dataset if no trained
artifacts are found, so it works out of the box either way).

---

## Project structure

```
emergency-prioritization/
├── app/                  FastAPI backend
│   ├── api/               routes: auth, messages, evaluation
│   ├── models/             SQLAlchemy models, seed data, Pydantic schemas
│   ├── services/           auth, priority scoring, RL inference bridge
│   ├── templates/          Jinja2 + Bootstrap dashboard, login page
│   ├── static/              CSS/JS
│   └── config.py
├── nlp/                   NLP pipeline (Stages 2-6)
│   ├── data_acquisition.py    downloads the real public dataset
│   ├── synthetic_data.py      generates the synthetic fallback dataset
│   ├── preprocessing.py       shared cleaning/tokenization
│   ├── classifier.py          TF-IDF + LogReg/SVM category classifier
│   ├── urgency.py              urgency classifier + disclosed heuristic
│   ├── ner.py                   location + assistance-type extraction
│   ├── duplicate_detection.py    TF-IDF cosine similarity
│   └── pipeline.py               unifies the above
├── rl/                    RL pipeline (Stages 7-8)
│   ├── environment.py        Gymnasium env: state/action/reward
│   ├── train.py                DQN training (Stable-Baselines3)
│   └── baseline.py              rule-based baseline
├── evaluation/             Stage 9: baseline vs RL comparison
├── streamlit_app/          Lightweight demo UI (reuses nlp/ and rl/)
├── tests/                  Automated tests
├── data/                   raw/ (real dataset), processed/, seed/ (synthetic)
├── docs/                   Academic documentation (see below)
├── setup.sh                One-command local setup
├── requirements.txt, .env.example, .gitignore
```

## Academic documentation

See the `docs/` folder for material written specifically for your project
report and viva:

- `docs/architecture.md` — system architecture, data flow
- `docs/nlp_workflow.md` — NLP pipeline details, model choices, metrics
- `docs/rl_workflow.md` — state/action/reward/policy explanation
- `docs/database_schema.md` — ER description
- `docs/api_reference.md` — endpoint summary
- `docs/evaluation_methodology.md` — how baseline vs RL is compared
- `docs/limitations.md`
- `docs/ethical_considerations.md`
- `docs/future_enhancements.md`
- `docs/viva_questions.md` — sample viva Q&A

## Dataset

- **Real data**: `nlp/data_acquisition.py` downloads the public
  disaster-response messages dataset (Figure Eight/Appen, ~26k
  human-labeled messages, mirrored at
  `github.com/canaveensetia/udacity-disaster-response-pipeline`), saved to
  `data/raw/`. It has **no urgency label** — see `nlp/urgency.py` for the
  disclosed heuristic used to bootstrap one if you train on this data.
- **Synthetic fallback**: `nlp/synthetic_data.py`, already generated at
  `data/seed/synthetic_messages.csv` (1200 records, `source=synthetic`
  tagged), used as the default training set so the project runs without
  internet access.

## License

For academic use. No warranty. Do not use for real emergency dispatch
without independent safety review, human oversight, and regulatory
compliance appropriate to your jurisdiction.
