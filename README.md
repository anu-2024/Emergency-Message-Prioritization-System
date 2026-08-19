# 🛰️ Emergency Message Prioritization System

**NLP + Reinforcement Learning Based Emergency Message Prioritization System**
*An MCA Academic Project — human-in-the-loop decision support only. This
system never autonomously dispatches emergency services.*

---

## What this is

Free-text emergency messages come in. Classical, CPU-friendly NLP classifies
each one (category, urgency, mentioned locations, requested assistance type,
likely duplicates). A Reinforcement Learning agent learns, through trial and
error against a simulated queue, *which* pending message a capacity-
constrained human responder should act on next — and its performance is
compared against a transparent, explainable rule-based baseline. Everything
is surfaced in a Streamlit dashboard where a human can always override the
suggested priority.

Built to the project's SRS (`SRS_Emergency_Message_Prioritization_System.docx`).

## Project structure

```
.
├── app.py                    # Streamlit front-end (run this to launch the UI)
├── rl_agent.py                # RL agent training/loading/evaluation script
├── nlp_training.ipynb         # NLP pipeline: EDA, training, model persistence
├── train_all.py                # Optional: pretrain NLP + RL in one command
├── requirements.txt
├── .streamlit/config.toml     # Dark "dispatch console" theme
├── src/
│   ├── config.py               # Shared constants & artifact paths
│   ├── data/
│   │   └── synthetic_data.py    # Real-dataset attempt + synthetic fallback (tagged provenance)
│   ├── nlp/
│   │   ├── preprocessing.py     # Cleaning + lemmatization (spaCy, with fallback)
│   │   ├── classifier.py        # Category + Urgency TF-IDF/LogReg classifiers
│   │   ├── ner.py               # Location NER + assistance-type keyword matcher
│   │   └── duplicate.py         # TF-IDF cosine-similarity duplicate detector
│   └── rl/
│       ├── environment.py       # Gymnasium-compatible triage environment
│       ├── baseline.py          # Transparent rule-based priority formula
│       └── evaluate.py          # Random vs Rule-Based vs RL evaluation
├── data/messages.csv           # Generated dataset (created on first run)
└── models/                     # Trained artifacts (created on first run)
```

The NLP and RL layers in `src/` have **no dependency on Streamlit** — they're
plain Python, importable from the notebook, `rl_agent.py`, a future FastAPI
backend, or a test suite, per the SRS's layered architecture.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional but recommended: pretrain everything once, so the app loads
# instantly instead of training on first visit.
python train_all.py

streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

If you skip `train_all.py`, the app still works: the NLP models train
in a couple of seconds on first use, and the RL agent trains a smaller
policy in the browser on first visit to **RL Training Lab** (with a live
learning-curve chart) — both are cached afterwards.

## Deploying on GitHub + Streamlit Community Cloud

1. `git init && git add -A && git commit -m "Emergency message prioritization system"`
2. Push to a new GitHub repository.
3. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at your repo, branch, and `app.py`.
4. Deploy.

**Recommended:** run `python train_all.py` locally first and commit the
resulting `models/` and `data/` folders (they are *not* gitignored on
purpose). That way the deployed app loads pretrained models immediately
instead of spending its first visitor's page-load training. If you skip
this, the app trains lazily and caches the result — it will still work,
just slower on the very first interaction with each page.

> **Resource note:** `stable-baselines3` + `torch` + `spacy` add real
> install weight. `requirements.txt` pins CPU-only PyTorch wheels
> (`--extra-index-url https://download.pytorch.org/whl/cpu`) to keep this
> as light as reasonably possible. If your deployment target has tight
> memory limits, the app degrades gracefully — NLP features keep working
> from cached `scikit-learn` models even if the RL page is slow to (re)train.

## The NLP notebook (`nlp_training.ipynb`)

Run it top-to-bottom with Jupyter (`jupyter notebook nlp_training.ipynb`) or
JupyterLab. It loads the dataset, does a short EDA, trains and evaluates the
category and urgency classifiers, **saves them to `models/`**, then reloads
them from disk to prove they work without retraining — the exact same
artifacts `app.py` consumes. It also demonstrates NER, the assistance-type
keyword matcher, and duplicate detection.

## The RL agent (`rl_agent.py`)

```bash
python rl_agent.py                        # train (default 40,000 timesteps), evaluate, save
python rl_agent.py --timesteps 100000      # longer run
python rl_agent.py --eval-only             # just re-evaluate an already-saved policy
```

This builds the Gymnasium environment (`src/rl/environment.py`), trains a
DQN policy with Stable-Baselines3 in chunks (logging a learning curve as it
goes), saves the policy to `models/dqn_policy.zip`, and runs the seeded
Random-vs-Rule-Based-vs-RL evaluation, saving results to
`models/eval_results.json`. `app.py`'s **RL Training Lab** page calls the
same `load_or_train_agent()` function interactively.

## What was executed & verified vs. written but not independently executed

Per SRS §10.2, disclosed here explicitly:

- **Executed and verified in development:** the full NLP pipeline (data
  generation, preprocessing, category/urgency classifier training +
  evaluation, NER + keyword matching, duplicate detection, model
  persistence and reload) — end to end, with real accuracy/F1 numbers
  produced against a held-out split.
- **Written to the same standard, execution depends on your environment:**
  the RL training loop (`rl_agent.py`, DQN via Stable-Baselines3) and the
  Streamlit UI (`app.py`) are built against the documented, stable
  Gymnasium / Stable-Baselines3 / Streamlit APIs, but were not executed in
  the authoring environment because those packages require internet access
  to install. **Run `python train_all.py` and `streamlit run app.py`
  yourself once after installing `requirements.txt`** to generate your own
  trained policy and confirm end-to-end behavior in your environment — the
  rule-based fallback, NLP pages, and app UI all work independently of
  whether the RL agent has been trained yet.

## Ethical & safety guardrails (SRS §10.1)

- Never autonomously triggers any real-world dispatch, notification, or
  emergency-service action.
- Every priority score is a suggestion only, subject to human review and
  override at all times.
- Location output is always labeled indicative, never verified geolocation.
- Urgency label provenance (keyword-tier heuristic) is always disclosed.
- **Academic demonstration only** — not for real emergency dispatch without
  independent safety review, human oversight, and regulatory compliance.

## Sample viva questions & answers

**Q: Why TF-IDF + Logistic Regression instead of a transformer model?**
A: SRS §2.4 requires CPU-only execution with no GPU dependency, and
explainability is an explicit academic requirement. Classical linear models
over TF-IDF features are fast to train, easy to explain (inspectable
coefficients), and sufficiently accurate for a well-separated 8-class
disaster-category problem.

**Q: Where do the urgency labels come from, since there's no public
ground-truth urgency dataset?**
A: A disclosed keyword-tier heuristic used during synthetic data generation
(`src/data/synthetic_data.py`). This provenance is surfaced everywhere
urgency appears in the UI — it is explicitly never presented as
human-verified ground truth (SRS §5.3).

**Q: How is this a genuine RL formulation and not "ranking rebranded as
RL"?**
A: `src/rl/environment.py` implements a real Gymnasium environment with
`reset()`/`step()`, a state that encodes per-slot + global features, a
discrete action space, and a reward function with four distinct signed
components (SRS §6.2–6.5). A DQN agent (Stable-Baselines3) is trained
against this environment via trial-and-error interaction, and its learned
policy is evaluated head-to-head against a random policy and a hand-written
rule-based policy on identical seeded episodes (SRS §6.8) — that comparison
is what demonstrates the agent actually learned something beyond the
hand-written heuristic.

**Q: What happens if the RL agent hasn't been trained yet when the app is
deployed?**
A: The rule-based baseline (SRS §6.7) is always computed and always
available as a transparent fallback — the system's core priority queue
never depends on the RL agent being trained (SRS §2.1).

**Q: What are the system's limitations?**
A: Synthetic, template-generated training data (not human-labeled real
disaster messages); heuristic (not verified) urgency labels; a simplified
simulated queueing environment rather than a validated real-world arrival
model; English-only; and location extraction is indicative NLP output, not
verified geolocation.

## Legal / licensing

Academic demonstration only. Must not be used for real emergency dispatch
without independent safety review, human oversight, and regulatory
compliance appropriate to the deploying jurisdiction.
