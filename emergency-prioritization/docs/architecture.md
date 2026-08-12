# System Architecture

## High-level data flow

```
[Responder/Public] --submits text--> [FastAPI /api/messages/submit]
                                            |
                                            v
                                    [NLP Pipeline]
                        ┌───────────────┬───────────────┬─────────────────┐
                        v               v               v                 v
                  [Category         [Urgency        [NER: location,   [Duplicate
                   Classifier]       Classifier]      assistance type]  Detection]
                        └───────────────┴───────────────┴─────────────────┘
                                            |
                                            v
                          [Priority Service: rule-based + RL agent]
                                            |
                                            v
                            [SQLite DB: Message + audit trail]
                                            |
                                            v
                    [Responder Dashboard: ranked queue, override, actions]
```

## Components

- **FastAPI backend** (`app/`): stateless HTTP API + server-rendered
  dashboard. Holds no ML logic itself — delegates to `nlp/` and `rl/`.
- **NLP pipeline** (`nlp/`): four independent, composable modules
  (category, urgency, NER, duplicates) unified by `nlp/pipeline.py`.
  Each module is trainable/testable in isolation.
- **RL pipeline** (`rl/`): a Gymnasium environment simulating the
  responder-capacity-constrained message stream, a DQN agent trained on
  it, and a rule-based baseline for comparison. The environment has zero
  dependency on the web app or database, so it can be trained and
  evaluated completely offline.
- **Database** (`app/models/database.py`): SQLite via SQLAlchemy. Stores
  messages, their NLP+RL outputs, users, and an append-only audit trail
  of every responder action (including overrides).
- **Dashboard** (`app/templates/`, `app/static/`): Bootstrap + vanilla JS,
  talks to the FastAPI JSON API via `fetch()`.
- **Streamlit demo** (`streamlit_app/`): a thin, single-user alternative
  front-end that imports the exact same `nlp/` and `rl/` modules — no
  logic duplication — intended for quick deployment/demo, not for the
  full multi-user, audited workflow.

## Why this split

Keeping `nlp/` and `rl/` free of any FastAPI/SQLAlchemy/Streamlit
imports means:
1. They can be unit-tested with zero web-framework dependencies (see
   `tests/test_nlp.py`, `tests/test_rl.py`, which only need
   scikit-learn/gymnasium, not fastapi).
2. The same trained models power both the full dashboard and the
   Streamlit demo without maintaining two implementations.
3. RL training (`rl/train.py`) can run as a long batch job independently
   of whether the web server is even running.

## Request lifecycle for one message

1. `POST /api/messages/submit` — text arrives with JWT-authenticated user.
2. `nlp/pipeline.analyze_message()` runs classification, urgency,
   NER, duplicate check (against an in-memory index rebuilt from the DB
   at startup) — all in one process, no external API calls.
3. Result is persisted as a `Message` row, status `nlp_processed`.
4. `app/services/priority_service.compute_final_priority()` computes the
   rule-based score (always) and, if a trained RL agent is available, an
   RL-derived score too. Status becomes `prioritized`.
5. Dashboard polls `GET /api/messages/queue`, which sorts by
   human-override > RL score > rule-based score.
6. A responder reviews in the modal, optionally overrides the priority or
   records an action (assign/escalate/resolve) — every such action is
   appended to `ResponderAction` (audit trail), never overwriting history.
