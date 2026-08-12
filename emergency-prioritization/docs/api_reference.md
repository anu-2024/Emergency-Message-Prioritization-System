# API Reference

Full interactive docs available at `/docs` (Swagger UI) when the app is
running. Summary below.

## Auth

### `POST /api/auth/login`
Form-encoded `username`, `password`. Returns `{access_token, token_type, role}`.
All other endpoints require `Authorization: Bearer <token>`.

## Messages

### `POST /api/messages/submit`
Body: `{"text": "...", "source": "web_form"}`.
Runs the full NLP pipeline, computes rule-based (+ RL if available)
priority, stores the message. Returns the full `MessageOut` object.

### `GET /api/messages/queue`
Returns all non-resolved/closed messages, sorted by effective priority
(human override > RL > rule-based), descending.

### `GET /api/messages/{message_id}`
Single message detail.

### `POST /api/messages/{message_id}/override`
Body: `{"new_priority": 0.0-1.0, "note": "optional"}`.
Human-in-the-loop override. Always audit-logged. Available to any
authenticated user (both admin and responder roles can override —
per SRS's emphasis on keeping a human in control at all times).

### `POST /api/messages/{message_id}/action`
Body: `{"action_type": "review|assign|escalate|resolve", "note": "optional"}`.
Updates message status and appends to the audit trail.

### `GET /api/messages/stats/summary`
Returns counts by status, urgency, category, and duplicate count — powers
the dashboard's stat cards.

## Evaluation

### `GET /api/evaluation/results`
Returns the latest `evaluation/results.json` (random / rule_based /
rl_agent comparison), or `{"available": false, ...}` if evaluation hasn't
been run yet.

## Notes

- All endpoints (except `/api/auth/login` and `/health`) require a valid
  JWT. Tokens expire after `ACCESS_TOKEN_EXPIRE_MINUTES` (default 60,
  configurable in `.env`).
- No endpoint autonomously dispatches emergency services — every write
  operation is a human-initiated review/override/status-change action.
