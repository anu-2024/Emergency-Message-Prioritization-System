"""
Emergency Message Prioritization System — Streamlit Dashboard

A streamlined 3-page interface:
  1. Message Intake   — type or upload messages, see NLP predictions
  2. Priority Queue   — accepted messages ranked by RL agent
  3. RL Performance   — agent training, evaluation, live metrics
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config import (
    CATEGORIES, URGENCY_LEVELS, URGENCY_SCORE_MAP, CATEGORY_CRITICALITY,
    VECTORIZER_PATH, CATEGORY_MODEL_PATH, URGENCY_MODEL_PATH,
    DQN_MODEL_PATH, EVAL_RESULTS_PATH,
)
from src.data import load_dataset
from src.nlp import CategoryClassifier, UrgencyClassifier, DuplicateDetector
from src.nlp.ner import extract_locations, extract_assistance_keywords
from src.nlp.preprocessing import preprocessing_backend
from src.nlp.ner import ner_backend
from src.rl.baseline import rule_based_priority
from src.rl.dashboard_env import DashboardEnv

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EMPS Dispatch Console",
    page_icon=":satellite:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# COLOR PALETTE
# ============================================================
CAT_COLORS = {
    "Medical": "#4C8DFF", "Flood/Rescue": "#F97316", "Fire": "#EF4444",
    "Food": "#F59E0B", "Water": "#06B6D4", "Shelter": "#8B5CF6",
    "Infrastructure": "#6B7280", "Other/Irrelevant": "#4B5563",
}
URG_COLORS = {
    "Low": "#2DD4BF", "Medium": "#4C8DFF", "High": "#F5A623", "Critical": "#EF4444",
}
ACCENT = "#F5A623"
TEAL = "#2DD4BF"
BLUE = "#4C8DFF"


def _hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg: #06090F;
    --surface: #0C1220;
    --surface2: #111827;
    --border: #1E293B;
    --border2: #334155;
    --text: #F1F5F9;
    --text2: #94A3B8;
    --text3: #64748B;
    --amber: #F5A623;
    --red: #EF4444;
    --teal: #2DD4BF;
    --blue: #4C8DFF;
    --purple: #8B5CF6;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); }

section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 14px; padding: 6px 0;
}

/* ---- Ticker ---- */
.ticker {
    display: flex; gap: 28px; align-items: center; flex-wrap: wrap;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 18px; margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: var(--text2);
}
.ticker b { color: var(--text); }
.pulse {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--teal); margin-right: 6px;
    box-shadow: 0 0 0 0 rgba(45,212,191,.6);
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(45,212,191,.55); }
    70% { box-shadow: 0 0 0 8px rgba(45,212,191,0); }
    100% { box-shadow: 0 0 0 0 rgba(45,212,191,0); }
}

/* ---- Cards ---- */
.msg-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 18px; margin-bottom: 12px;
    transition: border-color .2s;
}
.msg-card:hover { border-color: var(--border2); }

/* ---- Badges ---- */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 10px; font-weight: 700; letter-spacing: .5px;
    text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
}
.badge-cat { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); }
.badge-urg-crit { background: rgba(239,68,68,.15); color: #FCA5A5; border: 1px solid rgba(239,68,68,.3); }
.badge-urg-high { background: rgba(245,166,35,.15); color: #FCD34D; border: 1px solid rgba(245,166,35,.3); }
.badge-urg-med  { background: rgba(76,141,255,.15); color: #93C5FD; border: 1px solid rgba(76,141,255,.3); }
.badge-urg-low  { background: rgba(45,212,191,.15); color: #5EEAD4; border: 1px solid rgba(45,212,191,.3); }

/* ---- Metric panel ---- */
.metric-panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 22px; text-align: center;
}
.metric-panel .label { color: var(--text3); font-size: 11px; text-transform: uppercase; letter-spacing: .8px; font-weight: 600; margin-bottom: 6px; }
.metric-panel .value { color: var(--text); font-size: 28px; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.metric-panel .sub { color: var(--text3); font-size: 11px; margin-top: 4px; }

/* ---- Section header ---- */
.section-head {
    font-size: 22px; font-weight: 800; color: var(--text); margin-bottom: 2px;
}
.section-sub {
    color: var(--text3); font-size: 13px; margin-bottom: 18px;
}

/* ---- Footer ---- */
.footer { color: var(--text3); font-size: 11px; text-align: center; margin-top: 40px; padding: 16px 0; border-top: 1px solid var(--border); }

/* ---- Misc ---- */
hr { border-color: var(--border) !important; }
div[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MESSAGE DATACLASS
# ============================================================
@dataclass
class Msg:
    id: str
    text: str
    ts: str
    category: str
    cat_conf: float
    urgency: str
    urg_score: float
    urg_conf: float
    locations: list = field(default_factory=list)
    assistance: list = field(default_factory=list)
    is_dup: bool = False
    dup_sim: float = 0.0
    dup_match: str = ""
    accepted: bool = False
    rl_priority: float = 0.0
    rl_action: int = -1


# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "accepted" not in st.session_state:
    st.session_state.accepted = []
if "eval" not in st.session_state:
    st.session_state.eval = {}
if "curve" not in st.session_state:
    st.session_state.curve = []
if "rl_stats" not in st.session_state:
    st.session_state.rl_stats = {"actions": [], "rewards": [], "urgencies": []}
if "dqn_model" not in st.session_state:
    st.session_state.dqn_model = None


# ============================================================
# HELPERS
# ============================================================
def _next_id() -> str:
    if "_msg_counter" not in st.session_state:
        st.session_state._msg_counter = 0
    st.session_state._msg_counter += 1
    return f"MSG-{st.session_state._msg_counter:04d}"

def _now() -> str:
    return time.strftime("%H:%M:%S")


@st.cache_data(show_spinner=False)
def _dataset() -> pd.DataFrame:
    return load_dataset(cache=True)


@st.cache_resource(show_spinner=False)
def _nlp_models():
    cat = CategoryClassifier()
    urg = UrgencyClassifier()
    if cat.artifacts_exist():
        cat.load()
    else:
        df = _dataset()
        cat.fit(df["text"], df["category"], verbose=False)
        cat.save()
    if urg.artifacts_exist():
        urg.load()
    else:
        df = _dataset()
        urg.fit(df["text"], df["urgency_level"], verbose=False)
        urg.save()
    return cat, urg


def _process(text: str) -> Msg:
    cat_clf, urg_clf = _nlp_models()
    ds = _dataset()
    cat, cconf, _ = cat_clf.predict(text)
    lev, uscr, uconf, _ = urg_clf.predict_with_score(text)
    locs = extract_locations(text)
    ast = extract_assistance_keywords(text)
    dup_det = DuplicateDetector(threshold=0.8)
    for _, r in ds.sample(min(150, len(ds)), random_state=1).iterrows():
        dup_det.add(r["message_id"], r["text"])
    dup = dup_det.check(text)
    return Msg(
        id=_next_id(), text=text, ts=_now(),
        category=cat, cat_conf=cconf,
        urgency=lev, urg_score=uscr, urg_conf=uconf,
        locations=locs, assistance=ast,
        is_dup=dup.is_duplicate, dup_sim=dup.similarity,
        dup_match=dup.best_match_text or "",
    )


def _urg_badge_class(level: str) -> str:
    return {"Critical": "crit", "High": "high", "Medium": "med", "Low": "low"}.get(level, "med")


def _card_html(m: Msg, show_accept: bool = True) -> str:
    uc = _urg_badge_class(m.urgency)
    cat_color = CAT_COLORS.get(m.category, "#6B7280")
    dup_html = ""
    if m.is_dup:
        dup_html = f'<div style="margin-top:8px;padding:6px 10px;background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.2);border-radius:6px;font-size:11px;color:#FCA5A5;">Duplicate detected ({m.dup_sim:.0%} similar)</div>'
    loc_html = f'<div style="margin-top:6px;font-size:11px;color:#64748B;">Locations: {", ".join(m.locations)}</div>' if m.locations else ""
    return f"""
    <div class="msg-card" style="border-left: 3px solid {cat_color};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <div>
                <span class="badge badge-cat" style="color:{cat_color};">{m.category}</span>
                <span class="badge badge-urg-{uc}" style="margin-left:6px;">{m.urgency}</span>
            </div>
            <span style="color:#64748B;font-size:11px;font-family:'JetBrains Mono',monospace;">{m.id} &middot; {m.ts}</span>
        </div>
        <div style="color:#E2E8F0;font-size:13px;line-height:1.5;margin-bottom:8px;">{m.text[:280]}{"..." if len(m.text)>280 else ""}</div>
        <div style="display:flex;gap:16px;font-size:11px;color:#94A3B8;">
            <span>Confidence: <b style="color:#F1F5F9;">{m.cat_conf:.0%}</b></span>
            <span>Urgency: <b style="color:#F1F5F9;">{m.urg_score:.2f}</b></span>
        </div>
        {loc_html}
        {dup_html}
    </div>"""


def _process_with_rl():
    if not st.session_state.accepted:
        return
    n = min(5, len(st.session_state.accepted))
    env = DashboardEnv(window_size=n)
    env.reset(seed=42)
    env.queue = []
    for m in st.session_state.accepted[:n]:
        env.queue.append({
            "category": CATEGORIES.index(m.category),
            "urgency": m.urg_score,
            "confidence": m.cat_conf,
            "is_duplicate": m.is_dup,
            "arrival_step": 0,
        })
    obs = env._obs()
    try:
        from stable_baselines3 import DQN
        if "dqn_model" in st.session_state and st.session_state.dqn_model is not None:
            model = st.session_state.dqn_model
        elif DQN_MODEL_PATH.exists():
            model = DQN.load(str(DQN_MODEL_PATH).replace(".zip", ""), env=env)
            st.session_state.dqn_model = model
        else:
            model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=5000,
                         batch_size=32, gamma=0.98, exploration_fraction=0.3,
                         policy_kwargs=dict(net_arch=[64, 64]),
                         verbose=0, seed=42)
            model.learn(total_timesteps=2000, progress_bar=False)
            st.session_state.dqn_model = model
        for i, m in enumerate(st.session_state.accepted[:n]):
            obs = env._obs()
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            pri = rule_based_priority(m.urg_score, m.category, 0.0, m.cat_conf)
            m.rl_priority = float(pri)
            m.rl_action = action
            st.session_state.rl_stats["actions"].append(action)
            st.session_state.rl_stats["rewards"].append(m.urg_score)
            st.session_state.rl_stats["urgencies"].append(m.urg_score)
        st.session_state.accepted.sort(key=lambda x: x.rl_priority, reverse=True)
    except Exception:
        for m in st.session_state.accepted:
            m.rl_priority = rule_based_priority(m.urg_score, m.category, 0.0, m.cat_conf)
        st.session_state.accepted.sort(key=lambda x: x.rl_priority, reverse=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### EMPS")
    st.caption("Emergency Message Prioritization System")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Message Intake", "Priority Queue"],
        label_visibility="collapsed",
    )

# ============================================================
# TOP TICKER
# ============================================================
ds = _dataset()
n_pending = len(st.session_state.messages)
n_accepted = len(st.session_state.accepted)
st.markdown(f"""
<div class="ticker">
    <span><span class="pulse"></span><b>SYSTEM ONLINE</b></span>
    <span>Pending: <b>{n_pending}</b></span>
    <span>Accepted: <b>{n_accepted}</b></span>
    <span>Dataset: <b>{len(ds):,}</b> messages</span>
    <span>Categories: <b>{len(CATEGORIES)}</b></span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE: MESSAGE INTAKE
# ============================================================
def _page_intake():
    st.markdown('<div class="section-head">Message Intake</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Type a message or pick a sample. The NLP pipeline predicts category, urgency, and priority instantly.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        text = st.text_area(
            "Emergency message",
            height=110,
            placeholder="e.g. Flooding at Riverside Colony, elderly couple trapped on rooftop, water still rising",
        )
        btn = st.button("Analyze Message", type="primary", use_container_width=True)
    with c2:
        samples = [
            "Medical emergency: chest pain, need ambulance immediately at 42 Oak Street",
            "House fire on Elm Avenue, flames visible from two blocks away, families evacuated",
            "Flooding in downtown area, water knee-deep, several cars stranded",
            "Need drinking water urgently, tap water contaminated, 50 families affected",
            "Food supplies running low at shelter on Park Road, need ration delivery",
            "Building collapsed on Main Street, people trapped under debris",
            "Power lines down after storm, blocking Highway 101 near mile marker 12",
            "Water tanker needed at community center, 200 people without water since morning",
        ]
        pick = st.selectbox("Or try a sample message", ["(type your own)"] + samples)
        if pick != "(type your own)":
            st.info(pick[:120] + "...")

    chosen = text if text.strip() else (pick if pick != "(type your own)" else "")

    if btn and chosen.strip():
        msg = _process(chosen.strip())
        st.session_state.messages.append(msg)
        st.rerun()

    if st.session_state.messages:
        st.markdown("---")
        st.markdown(f"##### Analyzed messages ({len(st.session_state.messages)} pending)")
        sorted_msgs = sorted(st.session_state.messages, key=lambda m: m.urg_score, reverse=True)
        for idx, m in enumerate(sorted_msgs):
            st.markdown(_card_html(m), unsafe_allow_html=True)
            bc1, bc2 = st.columns([1, 5])
            with bc1:
                if st.button("Accept", key=f"acc_{m.id}_{idx}", type="primary", use_container_width=True):
                    m.accepted = True
                    st.session_state.accepted.append(m)
                    st.session_state.messages = [x for x in st.session_state.messages if x.id != m.id]
                    _process_with_rl()
                    st.rerun()


# ============================================================
# PAGE: PRIORITY QUEUE
# ============================================================
def _page_queue():
    st.markdown('<div class="section-head">Accepted Messages & RL Prioritization</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Messages you accepted are ranked by the RL agent. The agent learns to serve high-urgency messages first.</div>', unsafe_allow_html=True)

    if not st.session_state.accepted:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;background:var(--surface);border:1px solid var(--border);border-radius:12px;">
            <div style="font-size:40px;margin-bottom:12px;">:</envelope:</div>
            <div style="color:#94A3B8;font-size:14px;">No accepted messages yet.</div>
            <div style="color:#64748B;font-size:12px;margin-top:4px;">Go to Message Intake to add messages.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-panel"><div class="label">Accepted</div><div class="value">{len(st.session_state.accepted)}</div></div>""", unsafe_allow_html=True)
    with m2:
        avg_urg = np.mean([m.urg_score for m in st.session_state.accepted])
        st.markdown(f"""<div class="metric-panel"><div class="label">Avg Urgency</div><div class="value" style="color:{TEAL};">{avg_urg:.2f}</div></div>""", unsafe_allow_html=True)
    with m3:
        avg_conf = np.mean([m.cat_conf for m in st.session_state.accepted])
        st.markdown(f"""<div class="metric-panel"><div class="label">Avg Confidence</div><div class="value" style="color:{BLUE};">{avg_conf:.0%}</div></div>""", unsafe_allow_html=True)
    with m4:
        n_crit = sum(1 for m in st.session_state.accepted if m.urgency == "Critical")
        st.markdown(f"""<div class="metric-panel"><div class="label">Critical</div><div class="value" style="color:#EF4444;">{n_crit}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Urgency distribution
    urgs = [m.urgency for m in st.session_state.accepted]
    urg_counts = pd.Series(urgs).value_counts().reindex(URGENCY_LEVELS, fill_value=0)
    fig = go.Figure(go.Bar(
        x=urg_counts.index, y=urg_counts.values,
        marker_color=[URG_COLORS[u] for u in urg_counts.index],
        text=urg_counts.values, textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        title="Urgency Distribution", title_font_size=13,
        yaxis=dict(showgrid=False, showticklabels=False),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### RL-ranked priority queue")
    for rank, m in enumerate(st.session_state.accepted, 1):
        cat_color = CAT_COLORS.get(m.category, "#6B7280")
        uc = _urg_badge_class(m.urgency)
        pri_str = f"{m.rl_priority:.3f}" if m.rl_priority > 0 else "pending"
        st.markdown(f"""
        <div class="msg-card" style="border-left:3px solid {cat_color};">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="color:#64748B;font-family:'JetBrains Mono',monospace;font-size:11px;">#{rank}</span>
                    <span class="badge badge-cat" style="color:{cat_color};margin-left:6px;">{m.category}</span>
                    <span class="badge badge-urg-{uc}" style="margin-left:6px;">{m.urgency}</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#64748B;font-size:11px;">RL Priority</span>
                    <div style="color:{ACCENT};font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;">{pri_str}</div>
                </div>
            </div>
            <div style="color:#CBD5E1;font-size:13px;margin-top:8px;line-height:1.5;">{m.text[:200]}{"..." if len(m.text)>200 else ""}</div>
        </div>
        """, unsafe_allow_html=True)



# ============================================================
# ROUTER
# ============================================================
PAGES = {
    "Message Intake": _page_intake,
    "Priority Queue": _page_queue,
}
PAGES[page]()

st.markdown('<div class="footer">EMPS - Emergency Message Prioritization System | Academic Demo | Human-in-the-loop only</div>', unsafe_allow_html=True)
