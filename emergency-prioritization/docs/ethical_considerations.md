# Ethical Considerations

## Human-in-the-loop by design, not by policy alone

The system architecture enforces this, not just documentation:
- No API endpoint dispatches any external service. `POST /submit` only
  computes a suggested priority and stores it.
- Every priority score, RL or rule-based, is a *suggestion field* on the
  message record. Human responders act via explicit, audited actions
  (`assign`, `escalate`, `resolve`) — the system takes no autonomous
  action on their behalf.
- Overrides are always available and always take precedence over both
  RL and rule-based scores in the ranking.

## Bias and fairness risks

- The category and urgency classifiers were trained on template-generated
  synthetic text (or, if you retrain on the real dataset, on messages
  collected from specific real disaster events) — neither necessarily
  reflects the vocabulary, dialects, or reporting patterns of every
  population your deployment might serve. A classifier under-recognizing
  urgency in messages from underrepresented groups' phrasing patterns
  would be a serious real-world harm; this must be evaluated with
  demographically diverse validation data before any real use.
- The rule-based baseline's category-criticality weights
  (`rl/baseline.py: CATEGORY_CRITICALITY`) encode a value judgment about
  which emergency types matter more — this is a policy choice made by
  the system's designer (you), and should be reviewed by domain experts,
  not treated as objectively correct.

## Privacy

- Messages may contain personally identifying information (names,
  addresses, health conditions). This project stores them in a local
  SQLite file with no encryption at rest, and location extraction is
  explicitly disclosed as unverified NER output — do not treat it as
  precise geolocation. A real deployment needs data-retention policy,
  encryption at rest, and access controls beyond the basic RBAC here.

## Risk of automation bias

- Even with a human "in the loop," responders under time pressure may
  over-trust the displayed priority score. The dashboard surfaces both
  rule-based and RL scores side by side specifically so responders can
  see disagreement between the two methods rather than a single opaque
  number — disagreement is a signal to look closer, not average out.

## Dual-use / misuse considerations

- A prioritization system that under-serves certain message categories
  or locations, if deployed without oversight, could systematically
  delay help to those groups. This risk is why the SRS mandates and this
  implementation enforces human review before any real-world action.
