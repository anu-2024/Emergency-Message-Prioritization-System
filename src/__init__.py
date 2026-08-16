"""
Core, framework-agnostic modules for the Emergency Message Prioritization System.

This package is intentionally decoupled from any web/UI framework so that the
NLP and RL layers can be trained, tested and evaluated entirely offline, and
reused by any front-end (Streamlit demo, FastAPI dashboard, notebooks, CLI
scripts) that imports them. See SRS Section 9 (System Architecture).
"""
