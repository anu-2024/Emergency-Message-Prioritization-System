# Database Schema

SQLite via SQLAlchemy (`app/models/database.py`). Swap `DATABASE_URL` in
`.env` for a Postgres URL to migrate without code changes.

## Tables

### `users`
| Column | Type | Notes |
|---|---|---|
| id | PK int | |
| username | string, unique | |
| hashed_password | string | bcrypt hash, never plaintext |
| role | enum(admin, responder) | RBAC |
| full_name | string, nullable | |
| is_active | bool | soft-disable without deleting |
| created_at | datetime | |

### `messages`
Core entity — one row per submitted emergency message, covering its full
lifecycle (`status`: received → nlp_processed → prioritized → reviewed →
assigned/escalated → resolved → closed).

| Column | Type | Notes |
|---|---|---|
| id | PK int | |
| message_id | string, unique | public-facing ID, e.g. `MSG-A1B2C3` |
| raw_text | text | original submitted text |
| source | string | web_form / csv_import / seed_demo_data |
| status | enum | lifecycle stage |
| category | string | NLP output |
| category_confidence | float | |
| urgency | enum(Low/Medium/High/Critical) | NLP output |
| urgency_score | float | 0–1, mapped from urgency level |
| urgency_confidence | float | |
| locations | text (JSON list) | NER output, indicative only |
| assistance_types | text (JSON list) | keyword-matched |
| is_duplicate | bool | |
| duplicate_of_message_id | string, nullable | |
| duplicate_similarity | float, nullable | |
| rule_based_priority | float | always computed |
| rl_priority | float, nullable | null if RL model unavailable |
| final_priority_source | string | rule_based / rl / human_override |
| human_override_priority | float, nullable | |
| received_at | datetime | |
| updated_at | datetime | |

### `responder_actions`
Append-only audit trail. **Never updated or deleted** — every override,
review, assignment, escalation, or resolution is a new row.

| Column | Type | Notes |
|---|---|---|
| id | PK int | |
| message_id | FK → messages.id | |
| user_id | FK → users.id | who performed the action |
| action_type | string | review/assign/escalate/resolve/override |
| previous_value | text, nullable | |
| new_value | text, nullable | |
| note | text, nullable | |
| timestamp | datetime | |

### `evaluation_runs`
Historical snapshots of Stage 9 baseline-vs-RL comparisons, for the
dashboard's evaluation metrics view.

| Column | Type |
|---|---|
| id | PK int |
| run_at | datetime |
| policy_name | string (random / rule_based / rl_agent) |
| mean_reward, std_reward | float |
| mean_messages_served | float |
| mean_avg_served_urgency | float |
| n_episodes | int |

## Relationships

```
User 1---* ResponderAction *---1 Message
```

One user can perform many actions; one message can have many actions
(its full audit history); each action references exactly one user and
one message.
