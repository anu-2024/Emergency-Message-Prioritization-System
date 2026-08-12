# Future Enhancements

1. **Train on real historical response-time data.** Replace the
   heuristic/synthetic urgency labels with actual responder-assigned
   priority levels collected over time (the system already stores
   `human_override_priority` — this is a natural feedback-loop source).

2. **Reward shaping with domain-expert input.** Work with emergency
   management professionals to derive `rl/environment.py`'s reward
   weights from an actual triage protocol rather than hand-tuning.

3. **Larger RL formulation.** Move from a fixed 5-message window toward
   an attention-based policy (e.g. a Transformer or Pointer-Network
   policy) that can reason over the entire queue at once, if CPU budget
   allows during a later hardware upgrade.

4. **Online/continual learning.** Periodically retrain the urgency and RL
   models on newly collected, human-verified labels rather than a single
   offline training pass.

5. **Semantic duplicate detection.** Upgrade from TF-IDF cosine to a
   lightweight sentence-embedding model (e.g. a small distilled
   sentence-transformer) once CPU/latency budget allows, to catch
   duplicates phrased very differently.

6. **Multi-lingual support.** Extend `nlp/preprocessing.py` and retrain
   classifiers for languages relevant to your deployment region — the
   original Figure Eight dataset does include some multilingual data.

7. **Geocoding integration.** Pair the current NER location extraction
   with a real geocoding API (with appropriate rate limits and privacy
   review) to convert place names into map coordinates for dispatch
   planning — always kept advisory, never automatic.

8. **Horizontal scaling.** Migrate from SQLite to Postgres and add a
   task queue (e.g. Celery/Redis) for NLP inference if message volume
   grows beyond single-process capacity.

9. **A/B evaluation in a shadow-deployment.** Run the RL agent's
   suggestions alongside real responder decisions (without acting on
   them) to measure real-world agreement/disagreement before trusting it
   as the primary ranking signal.
