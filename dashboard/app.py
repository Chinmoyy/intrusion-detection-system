"""
Network Intrusion Detection System — Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""

import json
import os
import sys

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — dark cyberpunk theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #050d1a;
    color: #c8d8e8;
}

h1, h2, h3 {
    font-family: 'Share Tech Mono', monospace;
    color: #00ffe5;
    letter-spacing: 2px;
}

.stMetric {
    background: linear-gradient(135deg, #0a1628 0%, #0d2240 100%);
    border: 1px solid #00ffe520;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 0 20px #00ffe510;
}

.stMetric label { color: #7fa8c8 !important; }
.stMetric [data-testid="stMetricValue"] {
    color: #00ffe5 !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem !important;
}

.stSelectbox > div, .stNumberInput > div {
    background: #0a1628 !important;
    border: 1px solid #00ffe530 !important;
    color: #c8d8e8 !important;
}

.attack-badge {
    background: linear-gradient(90deg, #ff003c, #ff6b00);
    padding: 6px 16px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-weight: bold;
    letter-spacing: 2px;
    display: inline-block;
}
.normal-badge {
    background: linear-gradient(90deg, #00ffe5, #00b4ff);
    color: #050d1a;
    padding: 6px 16px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-weight: bold;
    letter-spacing: 2px;
    display: inline-block;
}

.sidebar .sidebar-content { background: #07101f; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load metrics
# ─────────────────────────────────────────────
@st.cache_data
def load_metrics():
    path = os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


meta = load_metrics()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("# 🛡️ NETWORK INTRUSION DETECTION SYSTEM")
st.markdown("##### ML-powered threat classification · NSL-KDD Dataset · Real-time scoring")
st.divider()

if meta is None:
    st.warning("⚠️ No trained model found. Run `python src/train.py` first.")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar — model selector
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ CONFIG")
    model_names = list(meta["model_results"].keys())
    selected_model = st.selectbox("Active Model", model_names,
                                  index=model_names.index(meta["best_model"])
                                  if meta["best_model"] in model_names else 0)
    st.caption(f"Best overall: **{meta['best_model']}**")
    st.divider()
    st.markdown("### 📂 Dataset")
    st.markdown("NSL-KDD (Network Security Lab)")
    st.markdown("41 features · 5 attack categories")
    st.markdown("[📥 Download Dataset](https://www.unb.ca/cic/datasets/nsl.html)")

# ─────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────
r = meta["model_results"][selected_model]
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Accuracy",  f"{r['accuracy']:.2%}")
col2.metric("⚡ F1 Score",  f"{r['f1_score']:.4f}")
col3.metric("📈 ROC AUC",   f"{r['roc_auc']:.4f}")
col4.metric("🧠 Model",     selected_model.split()[0])

st.divider()

# ─────────────────────────────────────────────
# Charts row
# ─────────────────────────────────────────────
left, right = st.columns([1, 1])

# Confusion matrix
with left:
    st.markdown("### Confusion Matrix")
    cm = np.array(r["confusion_matrix"])
    labels = ["Normal", "Attack"]
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        color_continuous_scale=[[0, "#050d1a"], [0.5, "#003355"], [1, "#00ffe5"]],
        labels=dict(x="Predicted", y="Actual"),
    )
    fig_cm.update_layout(
        paper_bgcolor="#050d1a", plot_bgcolor="#050d1a",
        font=dict(color="#c8d8e8", family="Share Tech Mono"),
        margin=dict(t=20, b=20),
        coloraxis_showscale=False,
    )
    fig_cm.update_xaxes(tickfont=dict(color="#00ffe5"))
    fig_cm.update_yaxes(tickfont=dict(color="#00ffe5"))
    st.plotly_chart(fig_cm, use_container_width=True)

# Model comparison bar chart
with right:
    st.markdown("### Model Comparison")
    models_list = list(meta["model_results"].keys())
    f1s  = [meta["model_results"][m]["f1_score"]  for m in models_list]
    aucs = [meta["model_results"][m]["roc_auc"]   for m in models_list]
    accs = [meta["model_results"][m]["accuracy"]  for m in models_list]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="F1 Score",  x=models_list, y=f1s,
                             marker_color="#00ffe5"))
    fig_bar.add_trace(go.Bar(name="ROC AUC",   x=models_list, y=aucs,
                             marker_color="#00b4ff"))
    fig_bar.add_trace(go.Bar(name="Accuracy",  x=models_list, y=accs,
                             marker_color="#ff6b00", opacity=0.7))
    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="#050d1a", plot_bgcolor="#050d1a",
        font=dict(color="#c8d8e8", family="Share Tech Mono"),
        legend=dict(bgcolor="#0a1628", bordercolor="#1a3a5c"),
        yaxis=dict(range=[0.8, 1.0], gridcolor="#0a2040"),
        xaxis=dict(gridcolor="#0a2040"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────
# Attack category breakdown
# ─────────────────────────────────────────────
st.divider()
st.markdown("### Attack Category Accuracy")

cat_data = meta.get("attack_category_results", {})
if cat_data:
    cat_df = pd.DataFrame([
        {"Category": k, "Samples": v["total"],
         "Correct": v["correct"], "Accuracy": v["accuracy"]}
        for k, v in cat_data.items()
    ]).sort_values("Accuracy", ascending=False)

    color_map = {
        "Normal": "#00ffe5", "DoS": "#ff003c",
        "Probe": "#ff6b00",  "R2L": "#bf00ff", "U2R": "#ffcc00",
        "Unknown": "#7fa8c8",
    }
    colors = [color_map.get(c, "#7fa8c8") for c in cat_df["Category"]]

    fig_cat = go.Figure(go.Bar(
        x=cat_df["Category"], y=cat_df["Accuracy"],
        marker_color=colors,
        text=[f"{a:.1%}" for a in cat_df["Accuracy"]],
        textposition="outside",
        customdata=cat_df[["Samples", "Correct"]].values,
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2%}<br>Samples: %{customdata[0]}<extra></extra>",
    ))
    fig_cat.update_layout(
        paper_bgcolor="#050d1a", plot_bgcolor="#050d1a",
        font=dict(color="#c8d8e8", family="Share Tech Mono"),
        yaxis=dict(range=[0, 1.15], gridcolor="#0a2040", tickformat=".0%"),
        xaxis=dict(gridcolor="#0a2040"),
        margin=dict(t=30, b=20),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

# ─────────────────────────────────────────────
# Live predictor
# ─────────────────────────────────────────────
st.divider()
st.markdown("### 🔍 Live Traffic Classifier")
st.caption("Fill in network connection features to classify in real-time.")

with st.expander("▶ Open Traffic Inspector", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    protocol  = c1.selectbox("Protocol",   ["tcp", "udp", "icmp"])
    service   = c2.selectbox("Service",    ["http", "ftp", "smtp", "ssh", "dns",
                                             "telnet", "https", "pop_3", "domain_u",
                                             "auth", "finger", "private", "other"])
    flag      = c3.selectbox("Flag",       ["SF", "S0", "REJ", "RSTO", "RSTR",
                                             "SH", "S1", "S2", "S3", "OTH"])
    logged_in = c4.selectbox("Logged In",  [1, 0])

    c5, c6, c7, c8 = st.columns(4)
    src_bytes  = c5.number_input("Src Bytes",   0, 1_000_000, 215)
    dst_bytes  = c6.number_input("Dst Bytes",   0, 1_000_000, 45076)
    duration   = c7.number_input("Duration (s)", 0, 60000,    0)
    count      = c8.number_input("Count",        0, 512,      1)

    if st.button("🔬 CLASSIFY TRAFFIC", type="primary"):
        try:
            from src.predict import IntrusionDetector
            detector = IntrusionDetector()

            record = {
                "duration": duration, "protocol_type": protocol,
                "service": service, "flag": flag,
                "src_bytes": src_bytes, "dst_bytes": dst_bytes,
                "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
                "num_failed_logins": 0, "logged_in": logged_in,
                "num_compromised": 0, "root_shell": 0, "su_attempted": 0,
                "num_root": 0, "num_file_creations": 0, "num_shells": 0,
                "num_access_files": 0, "num_outbound_cmds": 0,
                "is_host_login": 0, "is_guest_login": 0,
                "count": count, "srv_count": count,
                "serror_rate": 0.0, "srv_serror_rate": 0.0,
                "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
                "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
                "srv_diff_host_rate": 0.0, "dst_host_count": 255,
                "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
                "dst_host_diff_srv_rate": 0.0,
                "dst_host_same_src_port_rate": 0.0,
                "dst_host_srv_diff_host_rate": 0.0,
                "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
                "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
            }
            result = detector.predict(record)[0]
            badge = "attack-badge" if result["is_attack"] else "normal-badge"
            st.markdown(
                f'<span class="{badge}">{result["label"]}</span>'
                f'&nbsp;&nbsp; Confidence: <b>{result["confidence"]:.1%}</b>'
                f' &nbsp;|&nbsp; Attack prob: <b>{result["attack_prob"]:.1%}</b>',
                unsafe_allow_html=True,
            )
        except FileNotFoundError:
            st.warning("Train the model first: `python src/train.py`")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("Built with Python · Scikit-learn · Streamlit · Plotly  |  NSL-KDD Dataset  |  Baruch College CIS Portfolio")
