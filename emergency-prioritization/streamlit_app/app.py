"""
Lightweight Streamlit demo app.

Deliberately reuses the SAME nlp/ and rl/ modules as the FastAPI backend --
no duplicated NLP/RL logic. This is a quick-look, single-user demo UI
(easy to deploy to Streamlit Community Cloud), not a replacement for the
full FastAPI + Bootstrap dashboard, which has auth, RBAC, and a persistent
audit trail that Streamlit isn't well-suited for.

Run locally with:  streamlit run streamlit_app/app.py
Deploy: push this repo to GitHub, then on share.streamlit.io point the
app file at streamlit_app/app.py. Because model training artifacts
(nlp/artifacts/*.joblib) are gitignored, either:
  (a) remove those two lines from .gitignore before pushing so the
      trained models are included in the repo, or
  (b) add a one-time startup call to train the models on first boot
      (see `_ensure_models_trained()` below -- already wired in).
"""
import sys
from pathlib import Path

# Allow running `streamlit run streamlit_app/app.py` from repo root without
# needing the package installed -- adds the repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from nlp.pipeline import analyze_message
from nlp.duplicate_detection import DuplicateIndex
from rl.baseline import rule_based_priority

st.set_page_config(page_title="Emergency Message Prioritization — Demo", layout="wide")


def _ensure_models_trained():
    """First-run convenience: trains NLP models on the synthetic dataset
    if artifacts don't exist yet (e.g. fresh Streamlit Cloud deploy)."""
    from pathlib import Path as P
    artifacts = P(__file__).resolve().parent.parent / "nlp" / "artifacts"
    needed = ["category_classifier.joblib", "urgency_classifier.joblib"]
    if all((artifacts / f).exists() for f in needed):
        return
    with st.spinner("First run: training NLP models on the synthetic dataset (~10s)..."):
        from nlp.synthetic_data import generate, save_csv, OUTPUT_FILE
        if not OUTPUT_FILE.exists():
            save_csv(generate(1200), OUTPUT_FILE)
        from nlp.classifier import train as train_classifier
        from nlp.urgency import train as train_urgency
        train_classifier(OUTPUT_FILE)
        train_urgency(OUTPUT_FILE)


@st.cache_resource
def _init():
    _ensure_models_trained()
    return DuplicateIndex()


dup_index = _init()

st.title("🚨 Emergency Message Prioritization — Demo")
st.caption("NLP + Reinforcement Learning based prioritization | "
           "**Human-in-the-loop decision support — this demo never autonomously dispatches anything.**")

if "queue" not in st.session_state:
    st.session_state.queue = []

tab1, tab2, tab3 = st.tabs(["Submit & Analyze", "Priority Queue", "About this demo"])

with tab1:
    st.subheader("Submit a message")
    text = st.text_area("Emergency message text", height=100,
                         placeholder="e.g. Our street near Koramangala is flooded, two elderly people need urgent medical help")
    if st.button("Analyze", type="primary") and text.strip():
        msg_id = f"DEMO-{len(st.session_state.queue) + 1}"
        result = analyze_message(text, dup_index, msg_id)
        result["message_id"] = msg_id
        result["text"] = text
        result["rule_based_priority"] = rule_based_priority(
            urgency_score=result["urgency_score"], category=result["category"],
            waiting_time=0, category_confidence=result["category_confidence"],
            is_duplicate=result["is_duplicate"],
        )
        st.session_state.queue.append(result)

        col1, col2, col3 = st.columns(3)
        col1.metric("Category", result["category"], f"{result['category_confidence']*100:.1f}% confidence")
        col2.metric("Urgency", result["urgency"], f"{result['urgency_confidence']*100:.1f}% confidence")
        col3.metric("Rule-based priority", f"{result['rule_based_priority']:.3f}")

        if result["locations"]:
            st.info(f"📍 Detected location(s): {', '.join(result['locations'])} "
                    f"— *indicative only, verify before dispatch*")
        if result["assistance_types"]:
            st.info(f"🆘 Assistance type(s): {', '.join(result['assistance_types'])}")
        if result["is_duplicate"]:
            st.warning(f"⚠️ Possible duplicate of {result['duplicate_of']} "
                       f"(similarity: {result['duplicate_similarity']:.2f})")

with tab2:
    st.subheader("Current priority queue (this session)")
    if not st.session_state.queue:
        st.write("No messages submitted yet in this session.")
    else:
        df = pd.DataFrame(st.session_state.queue)
        df = df.sort_values("rule_based_priority", ascending=False)
        display_df = df[["message_id", "rule_based_priority", "urgency", "category",
                          "text", "is_duplicate"]].rename(columns={
            "message_id": "ID", "rule_based_priority": "Priority", "urgency": "Urgency",
            "category": "Category", "text": "Message", "is_duplicate": "Duplicate?",
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        if st.button("Clear queue"):
            st.session_state.queue = []
            st.rerun()

with tab3:
    st.markdown("""
    **This is a lightweight, single-user demo** of the NLP pipeline (category
    classification, urgency detection, NER, duplicate detection) and the
    rule-based prioritization baseline, sharing the exact same code as the
    full FastAPI backend (`nlp/` and `rl/` modules — nothing is duplicated
    or re-implemented here).

    **Not included in this Streamlit demo** (available in the full app):
    - Login / role-based access control (admin vs responder)
    - Persistent database + audit trail of responder actions
    - The trained RL agent's live prioritization (this demo shows the
      rule-based baseline only, to keep the deploy dependency-light —
      loading a Torch model on Streamlit Cloud's free tier is slow)
    - Multi-user shared queue (this queue is per-browser-session only)

    Run the full system locally with `uvicorn app.main:app --reload` for
    the complete experience described in the SRS.
    """)
