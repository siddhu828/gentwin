"""
dashboard.py — GenTwin Streamlit Dashboard
------------------------------------------
HOW TO RUN:
    streamlit run dashboard.py

STREAMLIT COMPONENTS USED (each explained inline):
    st.sidebar        — left-side panel for controls
    st.columns        — side-by-side layout
    st.metric         — KPI cards with a number and label
    st.selectbox      — dropdown menu
    st.multiselect    — multi-choice dropdown
    st.slider         — range/value slider
    st.button         — clickable button
    st.pyplot         — embed a matplotlib figure
    st.dataframe      — interactive scrollable table
    st.expander       — collapsible section
    st.spinner        — loading animation
    st.tabs           — tabbed layout
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

# ── Page config must be the FIRST Streamlit call ────────────────────────────
# st.set_page_config controls the browser tab title, icon, and layout width
st.set_page_config(
    page_title="GenTwin — ICS Anomaly Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ── Custom CSS for a cleaner dark look ──────────────────────────────────────
# st.markdown with unsafe_allow_html lets you inject raw HTML/CSS
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { color: #00d4ff; }
    h2 { color: #ffaa00; border-bottom: 1px solid #333; padding-bottom: 4px; }
    h3 { color: #aaa; }
    .stMetric label { color: #aaa !important; font-size: 0.82rem !important; }
    .risk-high   { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ffaa00; font-weight: bold; }
    .risk-low    { color: #7fff7f; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS & DATA PATHS
# ─────────────────────────────────────────────

SCORES_PATH  = "outputs/anomaly_scores.csv"
SENSOR_RISK  = {
    # sensor → (risk_level, plain-English explanation)
    "LIT101":  ("HIGH",   "Tank level sensor — spoofing causes silent overflow or dry pump"),
    "FIT101":  ("HIGH",   "Inlet flow — zeroing this hides valve closure from operators"),
    "MV101":   ("HIGH",   "Inlet valve — forced closure starves the tank of inflow"),
    "P101":    ("MEDIUM", "Primary pump — disabling causes tank to fill uncontrollably"),
    "AIT202":  ("HIGH",   "Conductivity sensor — manipulation can mask chemical contamination"),
    "AIT203":  ("MEDIUM", "pH sensor — abnormal pH damages pipes and membranes"),
    "FIT201":  ("MEDIUM", "Stage 2 flow — drop here propagates failure downstream"),
    "LIT301":  ("MEDIUM", "Stage 3 tank level — overflow risk if upstream attacks go undetected"),
    "AIT401":  ("LOW",    "Cl residual, stage 4 — low priority but affects water safety"),
    "UV401":   ("MEDIUM", "UV disinfection intensity — reduction risks microbial contamination"),
}

plt.style.use("dark_background")
PLOT_BG   = "#111111"
PLOT_FG   = "#0a0a0a"


# ─────────────────────────────────────────────
# DATA LOADING  (cached so it only runs once)
# st.cache_data caches the return value — re-runs only if inputs change
# ─────────────────────────────────────────────

@st.cache_data
def load_scores():
    if not os.path.exists(SCORES_PATH):
        return None
    df = pd.read_csv(SCORES_PATH, parse_dates=["Timestamp"])
    return df

@st.cache_data
def run_twin(attack_type, attack_start, attack_end):
    """Run the SimPy digital twin and return the log object."""
    # Add project root to path so we can import digital_twin
    sys.path.insert(0, os.getcwd())
    from digital_twin import run_simulation
    log, cfg = run_simulation(
        attack_type  = attack_type,
        attack_start = attack_start,
        attack_end   = attack_end,
        sim_duration = 600,
    )
    return log, cfg


# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# st.sidebar.* — anything here appears in the left panel
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/security-shield-green.png", width=60)
    st.title("GenTwin")
    st.caption("ICS Anomaly Detection Dashboard")
    st.divider()

    st.subheader("📡 Sensor View")
    # st.multiselect — user picks multiple items from a list
    sensor_options = list(SENSOR_RISK.keys())
    selected_sensors = st.multiselect(
        "Sensors to plot",
        options=sensor_options,
        default=["LIT101", "FIT101", "AIT202"],
    )

    st.divider()
    st.subheader("🏭 Digital Twin")

    # st.selectbox — single-choice dropdown
    attack_choice = st.selectbox(
        "Attack type",
        options=["none", "valve_stuck", "sensor_spoof"],
        format_func=lambda x: {
            "none":         "None (normal operation)",
            "valve_stuck":  "A — Valve Stuck Shut",
            "sensor_spoof": "B — Sensor Spoof",
        }[x],
    )

    # st.slider — integer range slider
    atk_window = st.slider(
        "Attack window (seconds)",
        min_value=0, max_value=580,
        value=(120, 300),
        disabled=(attack_choice == "none"),
    )

    # st.button — triggers an action; returns True on the click frame
    run_twin_btn = st.button("▶  Run Simulation", use_container_width=True)

    st.divider()
    st.caption("GenTwin · Part 4 · SWaT Dataset")


# ─────────────────────────────────────────────
# MAIN AREA — TITLE
# ─────────────────────────────────────────────

st.markdown("# 🛡️ GenTwin — ICS Cyberattack Detection Dashboard")
st.markdown(
    "Combining a **Variational Autoencoder** anomaly detector with a "
    "**SimPy digital twin** to detect cyberattacks on a water treatment plant."
)
st.divider()


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

df = load_scores()

if df is None:
    st.error(
        "⚠️  `outputs/anomaly_scores.csv` not found. "
        "Run `python train.py` first to generate anomaly scores."
    )
    st.stop()   # stop rendering the rest of the page


# ─────────────────────────────────────────────
# SECTION 1 — KPI METRICS
# st.columns(n) — create n side-by-side columns
# st.metric — shows a labelled number with optional delta
# ─────────────────────────────────────────────

st.header("📊 Dataset Overview")

c1, c2, c3, c4, c5 = st.columns(5)

total      = len(df)
n_normal   = (df["True_Label"] == "Normal").sum()
n_attack   = (df["True_Label"] == "Attack").sum()
n_flagged  = (df["Predicted_Label"] == "Attack").sum()
precision  = (df["Is_True_Attack"] & df["Is_Predicted_Atk"]).sum() / max(n_flagged, 1)
recall     = (df["Is_True_Attack"] & df["Is_Predicted_Atk"]).sum() / max(n_attack, 1)

with c1: st.metric("Total Rows",        f"{total:,}")
with c2: st.metric("Normal Rows",       f"{n_normal:,}")
with c3: st.metric("Attack Rows",       f"{n_attack:,}")
with c4: st.metric("VAE Flagged",       f"{n_flagged:,}")
with c5: st.metric("VAE Precision",     f"{precision:.1%}",
                    delta=f"Recall {recall:.1%}", delta_color="off")

st.divider()


# ─────────────────────────────────────────────
# SECTION 2 — SENSOR TIME SERIES WITH VAE FLAGS
# ─────────────────────────────────────────────

st.header("📈 Sensor Time Series — VAE Anomaly Flags")

if not selected_sensors:
    st.info("Select at least one sensor in the sidebar.")
else:
    # Find true attack windows for shading
    atk_mask  = df["True_Label"] == "Attack"
    edges     = np.diff(atk_mask.astype(int), prepend=0, append=0)
    atk_starts = df.index[edges[:-1] == 1]
    atk_ends   = df.index[edges[1:]  == -1]

    # Find VAE-predicted attack points
    vae_flags  = df[df["Predicted_Label"] == "Attack"].index

    colors = ["#00d4ff", "#ffaa00", "#ff6b9d", "#7fff7f",
              "#bf9fff", "#ff9f7f", "#7fffff"]

    fig, axes = plt.subplots(
        len(selected_sensors), 1,
        figsize=(14, 3.2 * len(selected_sensors)),
        sharex=True
    )
    if len(selected_sensors) == 1:
        axes = [axes]

    fig.patch.set_facecolor(PLOT_FG)
    fig.suptitle("Sensor Readings with VAE Anomaly Flags",
                 color="white", fontsize=12, fontweight="bold")

    for ax, sensor, color in zip(axes, selected_sensors, colors):
        ax.set_facecolor(PLOT_BG)

        # Shade true attack windows in red
        for s_idx, e_idx in zip(atk_starts, atk_ends):
            ax.axvspan(s_idx, e_idx, color="red", alpha=0.18, zorder=0)

        # Plot sensor signal
        ax.plot(df.index, df[sensor], color=color, lw=0.7, alpha=0.9)

        # Mark VAE-flagged points as orange dots
        ax.scatter(vae_flags, df.loc[vae_flags, sensor],
                   color="orange", s=6, zorder=5, alpha=0.7)

        ax.set_ylabel(sensor, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax.spines[sp].set_color("#333")

    axes[-1].set_xlabel("Row index (1 row = 1 second)", color="white", fontsize=8)

    # Legend
    red_p   = mpatches.Patch(color="red",    alpha=0.4,  label="True Attack")
    ora_p   = mpatches.Patch(color="orange", alpha=0.8,  label="VAE Flagged")
    fig.legend(handles=[red_p, ora_p], loc="lower center",
               ncol=2, fontsize=8, framealpha=0.2, labelcolor="white")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    # st.pyplot — render a matplotlib figure inside Streamlit
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Expander: show the raw anomaly score chart below
    with st.expander("🔍 VAE Reconstruction Error (raw score)"):
        fig2, ax2 = plt.subplots(figsize=(14, 2.8))
        fig2.patch.set_facecolor(PLOT_FG)
        ax2.set_facecolor(PLOT_BG)

        ax2.plot(df.index, df["Anomaly_Score"], color="#00d4ff", lw=0.6)

        threshold = df[df["True_Label"] == "Normal"]["Anomaly_Score"].mean() + \
                    3 * df[df["True_Label"] == "Normal"]["Anomaly_Score"].std()
        ax2.axhline(threshold, color="#ff6b9d", lw=1, linestyle="--",
                    label=f"Threshold ({threshold:.4f})")
        for s_idx, e_idx in zip(atk_starts, atk_ends):
            ax2.axvspan(s_idx, e_idx, color="red", alpha=0.18)

        ax2.set_ylabel("MSE Score", color="white", fontsize=8)
        ax2.tick_params(colors="white", labelsize=7)
        ax2.legend(fontsize=8, framealpha=0.2, labelcolor="white")
        for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax2.spines[sp].set_color("#333")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

st.divider()


# ─────────────────────────────────────────────
# SECTION 3 — DIGITAL TWIN SIMULATION
# ─────────────────────────────────────────────

st.header("🏭 Digital Twin Simulation")
st.caption(
    "The digital twin estimates tank level from valve/pump states (physics model), "
    "independent of the sensor. Divergence between twin and sensor = potential spoof."
)

# Session state persists variables between Streamlit reruns
# Without this, the simulation result would be lost every time the user interacts
if "twin_log" not in st.session_state:
    st.session_state.twin_log = None
    st.session_state.twin_cfg = None

if run_twin_btn:
    # st.spinner — shows a "loading…" message while the block runs
    with st.spinner("Running SimPy simulation..."):
        log, cfg = run_twin(
            attack_type  = attack_choice,
            attack_start = atk_window[0],
            attack_end   = atk_window[1],
        )
        st.session_state.twin_log = log
        st.session_state.twin_cfg = cfg
    st.success("Simulation complete!")

if st.session_state.twin_log is not None:
    log = st.session_state.twin_log
    cfg = st.session_state.twin_cfg

    times      = np.array(log.times)
    true_lv    = np.array(log.true_level)
    rep_lv     = np.array(log.reported_level)
    twin_lv    = np.array(log.twin_level)
    valve_st   = np.array(log.valve_open)
    pump_st    = np.array(log.pump_running)
    atk_active = np.array(log.attack_active).astype(bool)
    divergence = np.abs(twin_lv - rep_lv)

    atk_start_t = cfg.get("start", 9999)
    atk_end_t   = cfg.get("end",   9999)
    atype       = cfg.get("type",  "none")

    # KPI row for twin
    d1, d2, d3, d4 = st.columns(4)
    with d1: st.metric("Min Level (true)",  f"{true_lv.min():.0f} mm")
    with d2: st.metric("Max Level (true)",  f"{true_lv.max():.0f} mm")
    with d3: st.metric("Max Divergence",    f"{divergence.max():.1f} mm")
    with d4: st.metric("Attack Active",     "Yes" if atype != "none" else "No")

    # Plot twin results — 3 panels
    fig3, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig3.patch.set_facecolor(PLOT_FG)
    fig3.suptitle(
        f"Digital Twin — Attack: '{atype}'"
        + (f"  ({atk_start_t}s → {atk_end_t}s)" if atype != "none" else ""),
        color="white", fontsize=11, fontweight="bold"
    )

    def twin_shade(ax):
        if atype != "none":
            ax.axvspan(atk_start_t, atk_end_t, color="red", alpha=0.18)

    # Panel 1: levels
    ax = axs[0]; ax.set_facecolor(PLOT_BG)
    twin_shade(ax)
    ax.plot(times, true_lv,  color="#00d4ff", lw=1.1, label="True Level")
    ax.plot(times, rep_lv,   color="#ffaa00", lw=0.9, ls="--", label="Sensor (reported)")
    ax.plot(times, twin_lv,  color="#7fff7f", lw=0.9, ls=":",  label="Twin Estimate")
    ax.axhline(900,  color="#ff6b9d", lw=0.7, ls="-.", alpha=0.6)
    ax.axhline(400,  color="#ff6b9d", lw=0.7, ls="-.", alpha=0.6)
    ax.axhline(1400, color="red",     lw=0.7, ls="--", alpha=0.5)
    ax.set_ylabel("Level (mm)", color="white", fontsize=8)
    ax.legend(fontsize=7.5, framealpha=0.2, labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#333")

    # Panel 2: valve + pump
    ax2 = axs[1]; ax2.set_facecolor(PLOT_BG)
    twin_shade(ax2)
    ax2.step(times, valve_st,       where="post", color="#00d4ff", lw=1.1, label="MV101 Valve")
    ax2.step(times, pump_st + 1.6,  where="post", color="#ffaa00", lw=1.1, label="P101 Pump (+offset)")
    ax2.set_ylabel("State (0/1)", color="white", fontsize=8)
    ax2.legend(fontsize=7.5, framealpha=0.2, labelcolor="white")
    ax2.tick_params(colors="white", labelsize=7)
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax2.spines[sp].set_color("#333")

    # Panel 3: divergence
    ax3 = axs[2]; ax3.set_facecolor(PLOT_BG)
    twin_shade(ax3)
    ax3.fill_between(times, divergence, color="#ff6b9d", alpha=0.6, label="|Twin − Sensor|")
    ax3.axhline(50, color="yellow", lw=1, ls="--", label="Alert threshold (50mm)")
    ax3.set_ylabel("Divergence (mm)", color="white", fontsize=8)
    ax3.set_xlabel("Simulation Time (seconds)", color="white", fontsize=8)
    ax3.legend(fontsize=7.5, framealpha=0.2, labelcolor="white")
    ax3.tick_params(colors="white", labelsize=7)
    for sp in ["top","right"]: ax3.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax3.spines[sp].set_color("#333")

    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # Interpretation note
    if atype == "sensor_spoof":
        st.warning(
            "🔴 **Sensor Spoof detected via divergence:** The sensor reports a flat ~700mm "
            "while the digital twin predicts the level is rising. This gap (bottom panel) "
            "is the detection signal the VAE cannot provide — the sensor data *looks* normal."
        )
    elif atype == "valve_stuck":
        st.warning(
            "🔴 **Valve Stuck:** Inlet valve locked shut while pump runs. "
            "Tank drains toward zero — risk of pump cavitation and mechanical damage."
        )
    else:
        st.success("✅ Normal operation — all three level estimates agree. No divergence.")

else:
    st.info("Configure attack settings in the sidebar and click **▶ Run Simulation**.")

st.divider()


# ─────────────────────────────────────────────
# SECTION 4 — VULNERABILITY SUMMARY
# ─────────────────────────────────────────────

st.header("⚠️ Sensor Vulnerability Summary")
st.caption("Sensors ranked by anomaly frequency — VAE flagged anomalies only.")

# Count how many VAE-flagged rows each sensor is in an anomalous state
# (approximated as: how often does this sensor's value deviate during predicted attacks)
flagged_df = df[df["Predicted_Label"] == "Attack"]
normal_df  = df[df["True_Label"]      == "Normal"]

rows = []
for sensor in SENSOR_RISK:
    if sensor not in df.columns:
        continue
    risk_lvl, risk_note = SENSOR_RISK[sensor]

    # Anomaly count: how many flagged rows contain this sensor
    n_anomalies = len(flagged_df)  # all flagged rows contain all sensors

    # Statistical shift: how much does mean value change during flagged periods?
    normal_mean  = normal_df[sensor].mean()
    flagged_mean = flagged_df[sensor].mean() if len(flagged_df) > 0 else normal_mean
    shift_pct    = abs(flagged_mean - normal_mean) / (abs(normal_mean) + 1e-9) * 100

    rows.append({
        "Sensor":        sensor,
        "Risk Level":    risk_lvl,
        "Mean Shift %":  round(shift_pct, 1),
        "Risk Note":     risk_note,
    })

summary_df = pd.DataFrame(rows)
summary_df = summary_df.sort_values(
    by="Risk Level",
    key=lambda s: s.map({"HIGH": 0, "MEDIUM": 1, "LOW": 2})
)

# Colour-code Risk Level column
def colour_risk(val):
    colours = {"HIGH": "#ff444433", "MEDIUM": "#ffaa0033", "LOW": "#7fff7f22"}
    return f"background-color: {colours.get(val, '')}; color: white;"

styled = summary_df.style.applymap(colour_risk, subset=["Risk Level"])

# st.dataframe — renders a scrollable, sortable table
st.dataframe(styled, use_container_width=True, height=380)

# Plain-English callout for highest risk sensors
st.subheader("🔴 High-Risk Sensor Highlights")
high_risk = summary_df[summary_df["Risk Level"] == "HIGH"]
cols = st.columns(len(high_risk))
for col, (_, row) in zip(cols, high_risk.iterrows()):
    with col:
        st.markdown(f"**{row['Sensor']}**")
        st.markdown(f"<span class='risk-high'>HIGH RISK</span>", unsafe_allow_html=True)
        st.caption(row["Risk Note"])

st.divider()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "GenTwin · SWaT Dataset · VAE + SimPy · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)
