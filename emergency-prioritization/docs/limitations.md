# Limitations

Honest, disclosed limitations for the project report:

1. **Urgency labels are not human-annotated ground truth.** Neither the
   synthetic dataset nor the real Figure Eight dataset has genuine
   urgency annotations. Synthetic urgency is category-conditioned random
   assignment; real-data urgency (if you train on it) comes from a
   disclosed keyword heuristic. Any urgency-related accuracy numbers
   should be reported with this caveat.

2. **Category classifier's 100% test accuracy on synthetic data is
   inflated**, an artifact of template-based generation making categories
   very separable. Retrain on the real dataset for a realistic number.

3. **Location extraction is indicative, not verified.** spaCy's general
   NER was not fine-tuned on emergency-message text or on your local
   geography; false positives/negatives are expected. The system labels
   this in every response and expects human verification.

4. **RL training happens in a simulated environment**, not on live
   historical response data (none was available/specified). The learned
   policy's quality is bounded by how well rl/environment.py's synthetic
   message generator reflects real arrival patterns and true urgency
   distributions.

5. **Reward function weights are hand-tuned**, not derived from a formal
   clinical/disaster-response triage protocol. A real deployment should
   involve domain experts (emergency management professionals) in
   calibrating these.

6. **Small-window formulation.** The RL agent only ever sees 5 candidate
   messages at a time, not the entire queue. This keeps the problem
   tractable and CPU-trainable but means the agent cannot reason about
   messages outside its window except via the +1.0 global-max bonus term.

7. **Duplicate detection uses lexical similarity (TF-IDF cosine)**, not
   semantic embeddings -- it will miss duplicates phrased very differently
   but describing the same event, and may over-flag short generic
   messages that happen to share vocabulary.

8. **Single-machine, SQLite deployment.** Not tested under concurrent
   multi-responder load; a real deployment would need a proper database
   (Postgres) and likely a message queue for concurrent submissions.

9. **No production authentication hardening** (rate limiting, password
   complexity enforcement, refresh tokens, HTTPS termination) beyond
   basic JWT + bcrypt -- appropriate for an academic demo, not for a real
   emergency-services deployment without further security review.

10. **Build-environment constraint**: several components (FastAPI routes,
    DB models, RL training, Streamlit app) were written and syntax
    verified but could not be executed end-to-end in the development
    sandbox due to no internet access for package installation. See the
    README's "What was verified vs. not" table.
