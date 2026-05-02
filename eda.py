"""
eda.py — Exploratory Data Analysis for the SWaT Dataset
---------------------------------------------------------
WHAT THIS SCRIPT DOES:
    1. Loads the SWaT CSV (real or synthetic)
    2. Prints basic dataset statistics (shape, dtypes, missing values)
    3. Shows the Normal vs Attack class balance
    4. Plots 4 key sensors over time, with attack windows highlighted in RED

WHY WE DO EDA BEFORE MODELLING:
    "Train first, understand later" is the most common beginner mistake.
    EDA tells you:
      - Whether your data is clean (no missing values, correct dtypes)
      - How severe the class imbalance is (usually 90%+ normal in SWaT)
      - Whether attacks are actually *visible* in the raw signal
    If attacks aren't visible visually, your model will struggle too.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────
# CONFIGURATION — change paths here if needed
# ─────────────────────────────────────────────
DATA_PATH   = "data/swat_data.csv"
OUTPUT_DIR  = "outputs"             # plots will be saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sensors we'll plot — chosen because they represent different attack types
# and are the most physically meaningful sensors in Stage 1 & 2
SENSORS_TO_PLOT = ["LIT101", "FIT101", "AIT202", "FIT201"]

# Matching human-readable labels for the y-axis
SENSOR_LABELS = {
    "LIT101": "Tank 1 Level (mm)",
    "FIT101": "Inlet Flow Rate (L/s)",
    "AIT202": "Conductivity (μS/cm)",
    "FIT201": "Stage 2 Flow (L/s)",
}


# ─────────────────────────────────────────────
# STEP 1: Load the data
# ─────────────────────────────────────────────
print("=" * 60)
print("  GenTwin — SWaT Exploratory Data Analysis")
print("=" * 60)

print(f"\n📂 Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# The Timestamp column is a string — convert it to a proper datetime object
# so matplotlib can plot it on a time axis correctly
df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
df = df.set_index("Timestamp")      # set as index for easy time-series slicing

print(f"✅ Loaded successfully.\n")


# ─────────────────────────────────────────────
# STEP 2: Basic statistics
# ─────────────────────────────────────────────
print("─" * 40)
print("📐 SHAPE (rows × columns)")
print(f"   {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

print("\n📋 COLUMN NAMES")
# Print columns in groups of 6 for readability
cols = df.columns.tolist()
for i in range(0, len(cols), 6):
    print("   ", "  |  ".join(cols[i:i+6]))

print("\n🔍 DATA TYPES (first 10 columns)")
print(df.dtypes[:10].to_string())

print("\n❓ MISSING VALUES")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   No missing values! ✅")
else:
    print(missing[missing > 0])

print("\n📊 BASIC STATISTICS (Stage 1 sensors)")
# Show stats for just the most important sensors to avoid clutter
stage1_sensors = ["LIT101", "FIT101", "MV101", "P101"]
print(df[stage1_sensors].describe().round(3).to_string())


# ─────────────────────────────────────────────
# STEP 3: Class balance — Normal vs Attack
# ─────────────────────────────────────────────
print("\n─" * 40)
print("🏷️  LABEL DISTRIBUTION (Normal vs Attack)")

label_counts = df["Normal/Attack"].value_counts()
label_pct    = df["Normal/Attack"].value_counts(normalize=True) * 100

for label in label_counts.index:
    bar = "█" * int(label_pct[label] / 2)  # ASCII bar chart
    print(f"   {label:10s}: {label_counts[label]:>6,} rows  ({label_pct[label]:.1f}%)  {bar}")

# KEY INSIGHT: In real SWaT, attacks are ~12% of data.
# Class imbalance like this is WHY we train the VAE only on normal data —
# a standard classifier would just learn to always predict "Normal".

print("\n⚠️  Note: This imbalance is intentional — attacks are rare events.")
print("   This is exactly why we use an anomaly detector (VAE) instead of")
print("   a simple classifier.\n")


# ─────────────────────────────────────────────
# STEP 4: Find attack time windows
# ─────────────────────────────────────────────
# To highlight attacks in red on the plot, we need to find continuous
# stretches of "Attack" rows. We compute a boolean mask and find the
# start/end timestamps of each attack block.

attack_mask = df["Normal/Attack"] == "Attack"

# Find rising edges (Normal→Attack) and falling edges (Attack→Normal)
# np.diff gives us a +1 at the start of an attack, -1 at the end
edges = np.diff(attack_mask.astype(int), prepend=0, append=0)
attack_starts = df.index[edges[:-1] == 1]
attack_ends   = df.index[edges[1:]  == -1]   # one position after attack block

attack_windows = list(zip(attack_starts, attack_ends))

print(f"🔴 Found {len(attack_windows)} attack window(s):")
for i, (s, e) in enumerate(attack_windows):
    duration = (e - s).total_seconds()
    print(f"   Attack {i+1}: {s.strftime('%H:%M:%S')} → {e.strftime('%H:%M:%S')}  ({duration:.0f} seconds)")


# ─────────────────────────────────────────────
# STEP 5: Time-series plot with attack windows
# ─────────────────────────────────────────────
print("\n📈 Generating sensor time-series plots...")

# Use a clean dark style — easier to see attack highlights
plt.style.use("dark_background")

fig, axes = plt.subplots(
    nrows=len(SENSORS_TO_PLOT),
    ncols=1,
    figsize=(16, 12),            # wide and tall for 4 subplots
    sharex=True,                 # all plots share the same x-axis (time)
)

fig.suptitle(
    "SWaT Dataset — Key Sensor Readings Over Time\n(Red bands = injected attack windows)",
    fontsize=14,
    fontweight="bold",
    color="white",
    y=1.01
)

# Colour palette for each sensor line
LINE_COLORS = ["#00d4ff", "#ffaa00", "#ff6b9d", "#7fff7f"]

for ax, sensor, color in zip(axes, SENSORS_TO_PLOT, LINE_COLORS):

    # ── Plot the sensor signal ───────────────────────────────────────────
    ax.plot(
        df.index,
        df[sensor],
        color=color,
        linewidth=0.6,           # thin line so we can see detail
        alpha=0.9,
        label=sensor,
    )

    # ── Shade attack windows in semi-transparent red ─────────────────────
    for (start, end) in attack_windows:
        ax.axvspan(
            start, end,
            color="red",
            alpha=0.25,          # semi-transparent so signal is still visible
            label="_nolegend_",  # don't repeat "Attack" in the legend
        )

    # ── Y-axis label and formatting ──────────────────────────────────────
    ax.set_ylabel(SENSOR_LABELS.get(sensor, sensor), color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_facecolor("#111")

    # ── Mini-legend inside each subplot ─────────────────────────────────
    ax.legend(loc="upper right", fontsize=8, framealpha=0.3)

# ── Shared x-axis formatting ─────────────────────────────────────────────
axes[-1].set_xlabel("Time", color="white", fontsize=10)

# ── Add a single red patch to legend ─────────────────────────────────────
red_patch = mpatches.Patch(color="red", alpha=0.4, label="Attack Period")
fig.legend(handles=[red_patch], loc="lower center", ncol=1, fontsize=9,
           framealpha=0.2, labelcolor="white")

plt.tight_layout(rect=[0, 0.03, 1, 1])

# Save the figure
plot_path = os.path.join(OUTPUT_DIR, "sensor_overview.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
print(f"   Saved → {plot_path}")
plt.show()


# ─────────────────────────────────────────────
# STEP 6: Distribution plot — Normal vs Attack
# ─────────────────────────────────────────────
# This shows how much the VALUE DISTRIBUTION of each sensor shifts
# during attacks — a key intuition for why anomaly detection works.

print("\n📊 Generating distribution comparison plots...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle("Sensor Value Distributions — Normal vs Attack",
              fontsize=13, fontweight="bold", color="white")
fig2.patch.set_facecolor("#0a0a0a")

normal_df = df[df["Normal/Attack"] == "Normal"]
attack_df = df[df["Normal/Attack"] == "Attack"]

for ax, sensor in zip(axes2.flatten(), SENSORS_TO_PLOT):
    # Plot histogram for normal data
    ax.hist(normal_df[sensor].dropna(), bins=60, color="#00d4ff",
            alpha=0.6, label="Normal", density=True)
    # Overlay histogram for attack data
    ax.hist(attack_df[sensor].dropna(), bins=60, color="#ff4444",
            alpha=0.7, label="Attack",  density=True)

    ax.set_title(sensor, color="white", fontsize=10)
    ax.set_xlabel("Sensor Value", color="#aaa", fontsize=8)
    ax.set_ylabel("Density", color="#aaa", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_facecolor("#111")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")

plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, "sensor_distributions.png")
plt.savefig(dist_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
print(f"   Saved → {dist_path}")
plt.show()

# ─────────────────────────────────────────────
# STEP 7: Correlation heatmap (bonus insight)
# ─────────────────────────────────────────────
print("\n🔗 Generating correlation heatmap for Stage 1 & 2 sensors...")

# Correlation tells you: when one sensor goes up, do others tend to go up too?
# In a physical system, many sensors are correlated (e.g., if the pump is off,
# flow goes down AND tank level changes). Attacks break these correlations.

key_sensors = ["LIT101", "FIT101", "MV101", "P101",
               "AIT201", "AIT202", "AIT203", "FIT201"]

corr = df[key_sensors].corr()

fig3, ax3 = plt.subplots(figsize=(9, 7))
fig3.patch.set_facecolor("#0a0a0a")
ax3.set_facecolor("#111")

# Manual heatmap using imshow
im = ax3.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

# Add correlation value text inside each cell
for i in range(len(key_sensors)):
    for j in range(len(key_sensors)):
        text_color = "white" if abs(corr.values[i, j]) > 0.5 else "#aaa"
        ax3.text(j, i, f"{corr.values[i, j]:.2f}",
                 ha="center", va="center", fontsize=8, color=text_color)

ax3.set_xticks(range(len(key_sensors)))
ax3.set_yticks(range(len(key_sensors)))
ax3.set_xticklabels(key_sensors, rotation=45, ha="right", color="white", fontsize=9)
ax3.set_yticklabels(key_sensors, color="white", fontsize=9)
ax3.set_title("Sensor Correlation Matrix (Stage 1 & 2)", color="white", fontsize=12)

plt.colorbar(im, ax=ax3, fraction=0.046)
plt.tight_layout()

corr_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
plt.savefig(corr_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
print(f"   Saved → {corr_path}")
plt.show()

print("\n" + "=" * 60)
print("  ✅ EDA Complete! Check the outputs/ folder for your plots.")
print("=" * 60)
