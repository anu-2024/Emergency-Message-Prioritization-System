"""
app.py
======
Streamlit front-end for the NLP + Reinforcement Learning Based Emergency
Message Prioritization System (SRS 4.1: "an alternative lightweight
single-user demo interface exposing the NLP analysis pipeline for quick
demonstration purposes" — extended here to also visualize RL training and
the rule-based vs. RL evaluation comparison).

This file is intentionally UI-only: all NLP and RL logic lives in `src/`
and `rl_agent.py`, imported here and reused as-is (SRS Section 9,
Presentation Layer consuming the same backend logic as the FastAPI
dashboard would).

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud: point it at this file. On first launch,
if no pretrained RL policy is committed to the repo, the app trains a small
one automatically (cached) — see the "RL Training Lab" page.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import (
    CATEGORIES, URGENCY_LEVELS, URGENCY_SCORE_MAP, CATEGORY_CRITICALITY,
    VECTORIZER_PATH, CATEGORY_MODEL_PATH, URGENCY_MODEL_PATH,
)
from src.data import load_dataset
from src.nlp import CategoryClassifier, UrgencyClassifier, DuplicateDetector
from src.nlp.ner import extract_locations, extract_assistance_keywords, ner_backend
from src.nlp.preprocessing import preprocessing_backend
from src.rl.baseline import rule_based_priority
from src.rl.evaluate import load_eval_results, evaluate_all
from src.rl.environment import MessagePrioritizationEnv

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Emergency Message Prioritization | Dispatch Console",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme: dark "dispatch console" aesthetic
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg: #0B1220;
    --panel: #121A2C;
    --panel-alt: #16203570;
    --border: #24304A;
    --text: #E6EAF2;
    --text-dim: #8592AD;
    --amber: #F5A623;
    --red: #E5484D;
    --teal: #2DD4BF;
    --blue: #4C8DFF;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at 15% 0%, #101a30 0%, #0B1220 45%, #090d17 100%); }

section[data-testid="stSidebar"] {
    background: #0D1424;
    border-right: 1px solid var(--border);
}

/* Top status ticker */
.dispatch-ticker {
    display: flex; gap: 28px; align-items: center; flex-wrap: wrap;
    background: linear-gradient(90deg, #10182c, #121c33);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 18px; margin-bottom: 18px;
    font-family: 'JetBrains Mono', monospace; font-size: 12.5px; letter-spacing: .3px;
    color: var(--text-dim);
}
.dispatch-ticker b { color: var(--text); }
.pulse-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background: var(--teal); margin-right:6px;
    box-shadow: 0 0 0 0 rgba(45,212,191,.6); animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(45,212,191,.55); }
    70% { box-shadow: 0 0 0 8px rgba(45,212,191,0); }
    100% { box-shadow: 0 0 0 0 rgba(45,212,191,0); }
}

/* Headline */
.console-title { font-size: 28px; font-weight: 700; color: var(--text); margin-bottom: 2px; }
.console-sub { color: var(--text-dim); font-size: 14.5px; margin-bottom: 14px; }

/* Ethics banner */
.ethics-banner {
    border: 1px solid #3a2c10; background: #1c1608; color: #F5C97A;
    border-radius: 10px; padding: 10px 16px; font-size: 13px; margin-bottom: 18px;
    display:flex; gap:10px; align-items:center;
}

/* Cards */
.card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 20px; margin-bottom: 14px;
}
.card h4 { margin-top:0; color: var(--text); font-size: 15px; letter-spacing:.3px;
    text-transform: uppercase; color: var(--text-dim); font-weight:600; }

/* Priority queue row */
.queue-row {
    display:flex; align-items:center; gap:14px;
    background: var(--panel-alt); border-left: 4px solid var(--blue);
    border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
}
.queue-row.crit { border-left-color: var(--red); }
.queue-row.high { border-left-color: var(--amber); }
.queue-row.med  { border-left-color: var(--blue); }
.queue-row.low  { border-left-color: var(--teal); }

.badge {
    display:inline-block; padding: 2px 9px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing:.4px; text-transform:uppercase;
}
.badge-crit { background:#3a1216; color:#ff8188; border:1px solid #5c1c22; }
.badge-high { background:#3a2a0c; color:#ffc266; border:1px solid #5c4013; }
.badge-med  { background:#0c2340; color:#7fb2ff; border:1px solid #1d3a63; }
.badge-low  { background:#0c3330; color:#6fe3d4; border:1px solid #124f49; }

.footer-note { color: var(--text-dim); font-size: 12px; margin-top: 30px; text-align:center; }

hr { border-color: var(--border) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset(cache=True)


@st.cache_resource(show_spinner=False)
def get_nlp_models():
    """Load saved NLP models, training them on the fly (fast, CPU, classical
    ML) if the artifacts are not yet present on disk — guarantees the app
    works even on a completely fresh clone with no committed model files."""
    cat_clf = CategoryClassifier()
    urg_clf = UrgencyClassifier()

    if CategoryClassifier.artifacts_exist():
        cat_clf.load()
    else:
        df = get_dataset()
        cat_clf.fit(df["text"], df["category"], verbose=False)
        cat_clf.save()

    if UrgencyClassifier.artifacts_exist():
        urg_clf.load()
    else:
        df = get_dataset()
        urg_clf.fit(df["text"], df["urgency_level"], verbose=False)
        urg_clf.save()

    return cat_clf, urg_clf


def urgency_css_class(level: str) -> str:
    return {"Critical": "crit", "High": "high", "Medium": "med", "Low": "low"}.get(level, "med")


def badge_html(level: str) -> str:
    cls = urgency_css_class(level)
    return f'<span class="badge badge-{cls}">{level}</span>'


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛰️ DISPATCH CONSOLE")
    st.caption("NLP + RL Emergency Message Prioritization — Academic Demo")
    page = st.radio(
        "Navigate",
        [
            "Command Center",
            "Message Analyzer",
            "Priority Queue Simulator",
            "RL Training Lab",
            "About & Ethics",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Preprocessing backend:\n{preprocessing_backend()}")
    st.caption(f"NER backend:\n{ner_backend()}")
    st.divider()
    st.caption("⚠️ Human-in-the-loop only. This system never autonomously "
               "dispatches emergency services.")

# ---------------------------------------------------------------------------
# Shared top ticker
# ---------------------------------------------------------------------------
dataset = get_dataset()
st.markdown(
    f"""
    <div class="dispatch-ticker">
        <span><span class="pulse-dot"></span><b>SYSTEM ONLINE</b></span>
        <span>Dataset: <b>{len(dataset):,}</b> messages
            ({dataset['provenance'].value_counts().to_dict()})</span>
        <span>Categories tracked: <b>{len(CATEGORIES)}</b></span>
        <span>Mode: <b>Human-in-the-loop decision support</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# PAGE: Command Center
# ===========================================================================
def page_command_center():
    st.markdown('<div class="console-title">Command Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="console-sub">NLP + Reinforcement Learning Based Emergency '
        'Message Prioritization System — MCA academic project demo.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="ethics-banner">⚠️&nbsp; Every priority score shown in this '
        'application is a <b>suggestion only</b>, subject to human review and '
        'override at all times. Locations are indicative NLP output, not '
        'verified geolocation. No message ever triggers a real-world dispatch.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training messages", f"{len(dataset):,}")
    c2.metric("Categories", len(CATEGORIES))
    c3.metric("Urgency tiers", len(URGENCY_LEVELS))
    nlp_ready = CategoryClassifier.artifacts_exist() and UrgencyClassifier.artifacts_exist()
    c4.metric("NLP models", "Ready ✅" if nlp_ready else "Will train on first use")

    st.markdown("#### Category distribution in the training corpus")
    counts = dataset["category"].value_counts().reindex(CATEGORIES).fillna(0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color="#F5A623",
        marker_line_color="#1c1608", marker_line_width=1,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="messages", xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card"><h4>Architecture</h4>', unsafe_allow_html=True)
        st.markdown(
            "- **Presentation Layer** — this Streamlit demo (+ a Bootstrap/FastAPI "
            "dashboard in the full deployment)\n"
            "- **NLP Layer** — TF-IDF + Logistic Regression classifiers, "
            "spaCy/heuristic NER, TF-IDF cosine duplicate detection\n"
            "- **RL Layer** — Gymnasium environment + Stable-Baselines3 DQN, "
            "with a transparent rule-based baseline\n"
            "- **Persistence** — trained artifacts on disk (SQLite in the full app)"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card"><h4>Explicitly out of scope</h4>', unsafe_allow_html=True)
        st.markdown(
            "- Autonomous dispatch of emergency services or personnel\n"
            "- Precise, verified geolocation\n"
            "- Real-time telecom / SMS gateway integration\n"
            "- Multi-lingual support beyond English"
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ===========================================================================
# PAGE: Message Analyzer
# ===========================================================================
def page_message_analyzer():
    st.markdown('<div class="console-title">Message Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="console-sub">Run the saved NLP pipeline on a live message: '
        'category, urgency, entities, assistance type and duplicate check.</div>',
        unsafe_allow_html=True,
    )

    cat_clf, urg_clf = get_nlp_models()

    if "dup_detector" not in st.session_state:
        st.session_state.dup_detector = DuplicateDetector(threshold=0.8)
        # seed the detector with a slice of the training corpus so duplicate
        # checking has something realistic to compare against immediately
        seed_df = dataset.sample(n=min(150, len(dataset)), random_state=1)
        for _, row in seed_df.iterrows():
            st.session_state.dup_detector.add(row["message_id"], row["text"])

    example = st.selectbox(
        "Try an example, or write your own below",
        ["(write my own)"] + dataset.sample(6, random_state=7)["text"].tolist(),
    )
    default_text = "" if example == "(write my own)" else example
    text = st.text_area("Emergency message text", value=default_text, height=100,
                         placeholder="e.g. Flooding at Riverside Colony, elderly couple trapped on rooftop, water still rising")

    analyze = st.button("Analyze message", type="primary", use_container_width=False)

    if analyze and text.strip():
        category, cat_conf, cat_proba = cat_clf.predict(text)
        level, score, urg_conf, urg_proba = urg_clf.predict_with_score(text)
        locations = extract_locations(text)
        assistance = extract_assistance_keywords(text)
        dup = st.session_state.dup_detector.check(text)

        age_norm = 0.0  # a freshly submitted message has waited 0 by definition
        priority = rule_based_priority(score, category, age_norm, cat_conf)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="card"><h4>Category</h4>', unsafe_allow_html=True)
            st.markdown(f"### {category}")
            st.progress(cat_conf, text=f"confidence {cat_conf:.0%}")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card"><h4>Urgency</h4>', unsafe_allow_html=True)
            st.markdown(f"### {level}  {badge_html(level)}", unsafe_allow_html=True)
            st.progress(score, text=f"urgency score {score:.2f} / 1.00")
            st.caption("Label source: keyword-tier heuristic (disclosed — "
                       "not human-verified ground truth). See SRS §5.3.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card"><h4>Rule-based priority</h4>', unsafe_allow_html=True)
            st.markdown(f"### {priority:.2f}")
            st.progress(priority, text="0 (low) → 1 (critical)")
            st.caption("Transparent fallback score: 50% urgency + 25% category "
                       "criticality + 15% age + 10% confidence.")
            st.markdown("</div>", unsafe_allow_html=True)

        col4, col5 = st.columns(2)
        with col4:
            st.markdown('<div class="card"><h4>Extracted entities (indicative only)</h4>', unsafe_allow_html=True)
            st.write("**Locations mentioned:**", ", ".join(locations) if locations else "—")
            st.write("**Assistance type(s) requested:**",
                      ", ".join(assistance) if assistance else "—")
            st.caption("⚠️ Location output is indicative NLP extraction only, "
                       "never verified geolocation.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="card"><h4>Duplicate check</h4>', unsafe_allow_html=True)
            if dup.is_duplicate:
                st.error(f"⚠️ Likely duplicate — {dup.similarity:.0%} similar to a "
                          f"previously seen message (id {dup.best_match_id}). "
                          f"Flagged for human verification, not auto-discarded.")
                st.caption(f"Closest match: \u201c{dup.best_match_text}\u201d")
            else:
                st.success(f"No likely duplicate found (closest similarity "
                            f"{dup.similarity:.0%}, threshold {dup.threshold:.0%}).")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("##### Category probability breakdown")
        proba_df = pd.DataFrame({"category": list(cat_proba.keys()), "probability": list(cat_proba.values())})
        proba_df = proba_df.sort_values("probability", ascending=True)
        fig = go.Figure(go.Bar(
            x=proba_df["probability"], y=proba_df["category"], orientation="h",
            marker_color="#4C8DFF",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.session_state.dup_detector.add(f"live-{int(time.time())}", text)
    elif analyze:
        st.warning("Please enter a message to analyze.")


# ===========================================================================
# PAGE: Priority Queue Simulator
# ===========================================================================
def page_priority_queue():
    st.markdown('<div class="console-title">Priority Queue Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="console-sub">A batch of incoming messages, ranked by the '
        'rule-based baseline and (if trained) the RL agent side by side. Override '
        'any priority — every action is appended to the audit trail.</div>',
        unsafe_allow_html=True,
    )

    cat_clf, urg_clf = get_nlp_models()

    if "queue_batch" not in st.session_state:
        st.session_state.queue_batch = None
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []
    if "overrides" not in st.session_state:
        st.session_state.overrides = {}

    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        n_msgs = st.slider("Batch size", 3, 15, 6)
        if st.button("🔄 Pull new batch of incoming messages", type="primary"):
            sample = dataset.sample(n=n_msgs).reset_index(drop=True)
            rows = []
            rng = np.random.default_rng()
            for _, row in sample.iterrows():
                category, cat_conf, _ = cat_clf.predict(row["text"])
                level, score, urg_conf, _ = urg_clf.predict_with_score(row["text"])
                age = float(rng.integers(0, 45))  # simulated minutes waited
                age_norm = min(1.0, age / 60.0)
                priority = rule_based_priority(score, category, age_norm, cat_conf)
                rows.append({
                    "id": row["message_id"], "text": row["text"], "category": category,
                    "urgency_level": level, "urgency_score": round(score, 3),
                    "confidence": round(cat_conf, 3), "waited_min": age,
                    "rule_priority": round(priority, 3),
                })
            st.session_state.queue_batch = pd.DataFrame(rows).sort_values(
                "rule_priority", ascending=False).reset_index(drop=True)
            st.session_state.overrides = {}

    batch = st.session_state.queue_batch
    if batch is None:
        st.info("Click **Pull new batch of incoming messages** to populate the queue.")
        return

    st.markdown("##### Ranked priority queue (highest priority first)")
    for _, row in batch.iterrows():
        cls = urgency_css_class(row["urgency_level"])
        override = st.session_state.overrides.get(row["id"])
        eff_priority = override if override is not None else row["rule_priority"]
        override_note = " — <b>human override</b>" if override is not None else ""
        st.markdown(
            f"""
            <div class="queue-row {cls}">
                <div style="flex:1">
                    <b>#{row['id']}</b> · {row['category']} · {badge_html(row['urgency_level'])}
                    &nbsp;waited {int(row['waited_min'])}m
                    <div style="color:#8592AD; font-weight:400; margin-top:4px;">{row['text'][:110]}</div>
                </div>
                <div style="text-align:right; min-width:150px;">
                    priority <b>{eff_priority:.2f}</b>{override_note}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        oc1, oc2, oc3 = st.columns([2, 1, 1])
        with oc1:
            new_val = st.slider(f"Override priority for #{row['id']}", 0.0, 1.0,
                                 float(eff_priority), 0.01, key=f"slider_{row['id']}",
                                 label_visibility="collapsed")
        with oc2:
            if st.button("Apply override", key=f"apply_{row['id']}"):
                st.session_state.overrides[row["id"]] = new_val
                st.session_state.audit_log.append({
                    "time": time.strftime("%H:%M:%S"), "message_id": row["id"],
                    "action": "override", "detail": f"priority set to {new_val:.2f}",
                })
                st.rerun()
        with oc3:
            action = st.selectbox("Action", ["review", "assign", "escalate", "resolve"],
                                   key=f"action_{row['id']}", label_visibility="collapsed")
        if st.button(f"Record '{action}' for #{row['id']}", key=f"record_{row['id']}"):
            st.session_state.audit_log.append({
                "time": time.strftime("%H:%M:%S"), "message_id": row["id"],
                "action": action, "detail": "-",
            })
            st.success(f"Recorded '{action}' for message #{row['id']}.")

    st.markdown("##### Immutable audit trail (session)")
    if st.session_state.audit_log:
        st.dataframe(pd.DataFrame(st.session_state.audit_log)[::-1],
                      use_container_width=True, hide_index=True)
    else:
        st.caption("No actions recorded yet in this session.")


# ===========================================================================
# PAGE: RL Training Lab
# ===========================================================================
def page_rl_lab():
    st.markdown('<div class="console-title">RL Training Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="console-sub">Train (or load a cached) DQN agent and watch its '
        'learning curve, then compare it against the random and rule-based policies '
        'on identical seeded episodes.</div>',
        unsafe_allow_html=True,
    )

    try:
        import rl_agent
    except ImportError as e:
        st.error(f"RL dependencies are not installed in this environment: {e}. "
                  f"Make sure `gymnasium` and `stable-baselines3` are in requirements.txt.")
        return

    from src.config import DQN_MODEL_PATH

    model_cached = DQN_MODEL_PATH.exists()
    st.markdown(
        f'<div class="card"><h4>Agent status</h4>'
        f'Saved policy on disk: <b>{"Yes — models/dqn_policy.zip" if model_cached else "Not yet — will train on demand"}</b>'
        f'</div>', unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        timesteps = st.select_slider(
            "Training timesteps (per run)",
            options=[4000, 8000, 16000, 24000, 40000],
            value=8000 if not model_cached else 16000,
        )
    with col2:
        force_retrain = st.checkbox("Force retrain from scratch (ignore cached policy)")

    start = st.button("▶️ Train / Load agent and watch it learn", type="primary")

    if start:
        chunk_size = max(1000, timesteps // 6)
        progress_bar = st.progress(0.0, text="Starting...")
        chart_placeholder = st.empty()
        curve_so_far = []

        def _cb(chunk_idx, n_chunks, mean_reward):
            curve_so_far.append({"timesteps": chunk_idx * chunk_size, "mean_eval_reward": mean_reward})
            progress_bar.progress(chunk_idx / n_chunks,
                                   text=f"Chunk {chunk_idx}/{n_chunks} · mean eval reward {mean_reward:.2f}")
            fig = go.Figure(go.Scatter(
                x=[p["timesteps"] for p in curve_so_far],
                y=[p["mean_eval_reward"] for p in curve_so_far],
                mode="lines+markers", line=dict(color="#2DD4BF", width=3),
                marker=dict(size=7),
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=340, margin=dict(l=10, r=10, t=30, b=10),
                title="Live learning curve — mean evaluation reward per chunk",
                xaxis_title="training timesteps", yaxis_title="mean episode reward",
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        with st.spinner("Training DQN agent (CPU)... this can take a minute or two."):
            model, curve, was_cached = rl_agent.load_or_train_agent(
                total_timesteps=timesteps, chunk_size=chunk_size,
                progress_callback=_cb, force_retrain=force_retrain,
            )
        progress_bar.progress(1.0, text="Done.")

        if was_cached and not curve_so_far:
            st.success("Loaded a previously trained policy from disk (models/dqn_policy.zip).")
            if curve:
                fig = go.Figure(go.Scatter(
                    x=[p["timesteps"] for p in curve], y=[p["mean_eval_reward"] for p in curve],
                    mode="lines+markers", line=dict(color="#2DD4BF", width=3),
                ))
                fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=340, title="Learning curve from last training run",
                    xaxis_title="training timesteps", yaxis_title="mean episode reward",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success(f"Training complete over {timesteps:,} timesteps.")

        st.session_state.rl_model = model

        with st.spinner("Running seeded evaluation: Random vs Rule-Based vs RL..."):
            summaries = evaluate_all(sb3_model=model, n_episodes=25, save=True)
        st.session_state.eval_summaries = summaries

    summaries = st.session_state.get("eval_summaries") or load_eval_results()
    if summaries:
        st.markdown("##### Evaluation — identical seeded episodes")
        policies = list(summaries.keys())
        rewards = [summaries[p]["mean_episode_reward"] for p in policies]
        served = [summaries[p]["mean_messages_served"] for p in policies]
        urgency = [summaries[p]["mean_urgency_served"] for p in policies]

        colors = {"Random": "#8592AD", "Rule-Based": "#4C8DFF", "RL (DQN)": "#2DD4BF"}
        bar_colors = [colors.get(p, "#F5A623") for p in policies]

        m1, m2, m3 = st.columns(3)
        for col, metric_vals, title in [
            (m1, rewards, "Mean episode reward"),
            (m2, served, "Mean messages served"),
            (m3, urgency, "Mean urgency of served msgs"),
        ]:
            fig = go.Figure(go.Bar(x=policies, y=metric_vals, marker_color=bar_colors))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=280, margin=dict(l=10, r=10, t=30, b=10), title=title,
            )
            col.plotly_chart(fig, use_container_width=True)

        st.caption("All three policies are evaluated on identical seeded episodes "
                   "of the same Gymnasium environment (SRS §6.8).")
    else:
        st.info("Train or load an agent above to see the evaluation comparison.")


# ===========================================================================
# PAGE: About & Ethics
# ===========================================================================
def page_about():
    st.markdown('<div class="console-title">About & Ethics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="console-sub">Design rationale, limitations and the ethical '
        'guardrails this project deliberately builds in.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card"><h4>Ethical & safety requirements</h4>', unsafe_allow_html=True)
    st.markdown(
        "- The system **never** autonomously triggers any real-world dispatch, "
        "notification, or emergency-service action.\n"
        "- Every displayed priority score is a **suggestion only**, subject to "
        "human review and override at all times.\n"
        "- Location output is always labeled **indicative** and requiring "
        "human verification.\n"
        "- Urgency label provenance (synthetic keyword-tier heuristic) is "
        "always disclosed, never presented as unambiguous ground truth.\n"
        "- This is an **academic demonstration**. It must not be used for real "
        "emergency dispatch without independent safety review, human oversight, "
        "and regulatory compliance appropriate to the deploying jurisdiction."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h4>NLP subsystem</h4>', unsafe_allow_html=True)
        st.markdown(
            "- Preprocessing: lowercase, URL/mention strip, punctuation cleanup, "
            "spaCy lemmatization + stopword removal (fallback: stopword removal only)\n"
            "- Category: TF-IDF (1-2 grams) + Logistic Regression, 8 classes\n"
            "- Urgency: TF-IDF + Logistic Regression, 4 classes, "
            "keyword-tier heuristic labels (disclosed)\n"
            "- NER: spaCy `en_core_web_sm` for locations (fallback: capitalized-"
            "phrase heuristic), separate disclosed keyword matcher for assistance type\n"
            "- Duplicates: TF-IDF cosine similarity vs. prior messages, configurable threshold"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h4>RL subsystem</h4>', unsafe_allow_html=True)
        st.markdown(
            "- Environment: Gymnasium-compatible, bounded visible window, "
            "stochastic arrivals\n"
            "- State: per-slot one-hot category + urgency + confidence + "
            "wait time + duplicate flag, plus global queue-length/progress\n"
            "- Action: discrete, one per visible slot (empty slot = penalized no-op)\n"
            "- Reward: + high urgency served, − duplicate served, "
            "+ bonus for globally-highest-urgency served, − queue waiting time\n"
            "- Policy: DQN (Stable-Baselines3), lightweight MLP, CPU only"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><h4>Limitations (disclosed, not hidden)</h4>', unsafe_allow_html=True)
    st.markdown(
        "- Urgency labels are heuristic, not human-verified ground truth — "
        "no public dataset provides this.\n"
        "- The synthetic dataset is template-generated; a real deployment "
        "should be retrained on verified, human-labeled disaster messages.\n"
        "- The RL environment is a simplified simulation of arrival/triage "
        "dynamics, not a validated real-world queueing model.\n"
        "- English only; no verified geolocation; single-responder capacity model."
    )
    st.markdown("</div>", unsafe_allow_html=True)


PAGES = {
    "Command Center": page_command_center,
    "Message Analyzer": page_message_analyzer,
    "Priority Queue Simulator": page_priority_queue,
    "RL Training Lab": page_rl_lab,
    "About & Ethics": page_about,
}

PAGES[page]()

st.markdown(
    '<div class="footer-note">NLP + RL Emergency Message Prioritization System · '
    'Academic demo · Human-in-the-loop decision support only</div>',
    unsafe_allow_html=True,
)
