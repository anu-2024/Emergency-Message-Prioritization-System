# Sample Viva Questions and Answers

### Q1: Why does this project need both NLP and RL — couldn't you just use one?
NLP extracts *what* a message is about (category, urgency, location,
duplicates) — it's a perception/understanding task, evaluated once per
message, with a fixed right answer. RL solves a different problem:
*sequencing decisions under a resource constraint over time* — given
limited responder capacity, which message to serve next, right now, given
what's arrived so far and what's still waiting. NLP alone can't answer
"serve now or wait" because that depends on what else might arrive later
and what capacity is used elsewhere — a sequential decision problem, which
is exactly what RL formalizes (state, action, reward, policy learned
through interaction).

### Q2: What's the difference between your rule-based baseline and the RL agent — aren't they doing the same thing?
The rule-based baseline computes a fixed formula
(`0.5*urgency + 0.2*category + 0.2*age + 0.1*confidence`) independently
for each message, using only that message's own features. It never learns
and never considers other messages in the queue. The RL agent's policy is
learned through trial-and-error interaction with the environment,
conditioned on the *entire visible window* of messages plus global queue
state, and is explicitly rewarded (`+1.0` bonus) for picking the message
that is highest-urgency *across the whole queue*, not just within its
local view — something the rule-based formula structurally cannot do
since it never compares across messages.

### Q3: How do you know your RL agent actually learned something, rather than just being a black box?
Two ways: (1) `evaluation/compare_policies.py` runs it against a random
baseline and the rule-based baseline on identical seeded episodes — if it
doesn't clearly beat random, training failed. (2) The environment's
reward function was independently sanity-checked: a hand-coded
"always pick highest-visible-urgency" policy scores 64.7 mean reward vs.
26.4 for random over 50 steps, confirming the reward genuinely
distinguishes good from bad behavior before any learning algorithm is
even involved.

### Q4: Why TF-IDF + Logistic Regression instead of BERT/a transformer?
Per the project's CPU-only constraint and the SRS's explicit instruction
to start with classical baselines: TF-IDF+LogReg trains in under a second,
needs no GPU, is fully interpretable (you can inspect `model.coef_` for
which words drive each prediction), and gets reasonable accuracy on a
text classification task of this size and vocabulary. A transformer would
add training time, memory, and inference latency for likely modest
accuracy gains on a domain with fairly distinct vocabulary per category.

### Q5: Where do your urgency labels come from, since the real dataset doesn't have them?
Disclosed explicitly in `nlp/urgency.py`: for the synthetic dataset,
urgency is assigned by category-conditioned random weights at data
generation time. For the real Figure Eight dataset, a transparent
keyword-tier heuristic (`heuristic_urgency_label`) bootstraps a weak
label. This is a documented limitation — a real system should ideally use
labels from actual trained responders' historical decisions.

### Q6: What happens if the RL model isn't available (not trained yet)?
`app/services/priority_service.py` always computes the rule-based
priority first as a guaranteed fallback, and only attempts RL scoring on
top. If the RL model file doesn't exist or fails to load, it logs a
warning once and the system silently falls back to rule-based-only
ranking — the dashboard still works, just without the RL column filled
in.

### Q7: How do you prevent the system from ever autonomously dispatching help?
Architecturally: no code path in `app/api/` calls any external
dispatch/notification service. The only "actions" the system supports are
database status updates (`review`, `assign`, `escalate`, `resolve`)
initiated by an authenticated human via the dashboard, always logged to
the immutable `responder_actions` audit table. There is no automated
trigger anywhere in the codebase.

### Q8: What's your evaluation metric and why?
Primarily mean episode reward (directly reflects the reward function's
definition of good prioritization: serve high-urgency messages, avoid
duplicates, don't let the queue idle). Secondary metrics: messages served
per episode (throughput) and average urgency of served messages (are we
actually prioritizing the right things, not just clearing volume).

### Q9: What are the biggest weaknesses of this project, and how would you address them in a real deployment?
See `docs/limitations.md` in full — top three: (1) urgency labels aren't
real ground truth, (2) RL trains and evaluates in a simulated
environment rather than on real historical data, (3) duplicate detection
uses lexical (TF-IDF) rather than semantic similarity. All three are
addressable with more data and iteration, discussed in
`docs/future_enhancements.md`.

### Q10: Why SQLite instead of a "real" database?
SRS §18 explicitly specifies SQLite for the CPU-friendly, single-machine
academic deployment target. The code uses SQLAlchemy as an ORM
abstraction, so switching to Postgres for a production deployment is a
one-line `DATABASE_URL` change with no application code changes.

### Q11: How does duplicate detection work, and what are its limits?
TF-IDF vectorizes the candidate message alongside all previously seen
messages; cosine similarity above a threshold (0.65) flags a duplicate,
returning the best-matching prior message so a responder can visually
verify. Limitation: purely lexical — two messages describing the same
event in very different words (e.g. different language register) would
be missed. A semantic embedding model would catch more but costs more
CPU/latency, a deliberate tradeoff disclosed in the docs.

### Q12: What would you change about the reward function if you had more time?
Get real domain-expert input on relative weights (currently hand-tuned:
2.0 for urgency, 0.5 duplicate penalty, 1.0 global-max bonus, 0.3 no-op
penalty, 0.01 per-step waiting penalty) rather than my own estimates, and
validate the reward against real historical triage decisions rather than
only a simulated environment.
