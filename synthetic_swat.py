"""
synthetic_swat.py
-----------------
Generates a realistic fake version of the SWaT (Secure Water Treatment) dataset.

WHY THIS EXISTS:
    The real SWaT dataset requires signing an NDA with iTrust, Singapore.
    Until you get it, this synthetic version uses the EXACT same column names,
    timestamp format, and sensor ranges so your EDA and model code will work
    unchanged once you swap in the real data.

HOW IT WORKS:
    1. We simulate physical sensor behaviour (e.g., tank level rises when inlet
       valve is open, falls when outlet pump runs).
    2. We inject a few "attack windows" where sensor values behave abnormally —
       this is what your AI model will later learn to detect.

COLUMN NAME REFERENCE (matches real SWaT):
    Timestamp   - datetime string, one row per second
    FIT101      - Flow rate into tank 1 (L/s)       Stage 1
    LIT101      - Water level in tank 1 (mm)         Stage 1
    MV101       - Motor valve 101 (0=closed,1=open)  Stage 1
    P101        - Pump 101 on/off (0/1)              Stage 1
    P102        - Pump 102 on/off (0/1)              Stage 1
    AIT201      - Chemical concentration, stage 2     Stage 2
    AIT202      - Conductivity measurement             Stage 2
    AIT203      - pH level                            Stage 2
    FIT201      - Flow rate, stage 2 (L/s)           Stage 2
    MV201       - Motor valve 201                     Stage 2
    LIT301      - Water level in tank 3 (mm)         Stage 3
    DPIT301     - Differential pressure               Stage 3
    FIT301      - Flow rate, stage 3 (L/s)           Stage 3
    MV301/302   - Motor valves stage 3               Stage 3
    P301/P302   - Pumps stage 3                      Stage 3
    AIT401/402  - Chemical sensors stage 4            Stage 4
    FIT401      - Flow rate, stage 4 (L/s)           Stage 4
    P401-P404   - Pumps stage 4                      Stage 4
    UV401       - UV light intensity (disinfection)   Stage 4
    AIT501-503  - Chemical sensors stage 5            Stage 5
    FIT501-504  - Flow sensors stage 5                Stage 5
    P501/P502   - Pumps stage 5                      Stage 5
    PIT501-503  - Pressure sensors stage 5            Stage 5
    FIT601      - Final flow rate                    Stage 6
    P601-P603   - Pumps stage 6                      Stage 6
    Normal/Attack - Ground truth label
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Total duration of the simulation
TOTAL_SECONDS = 10_000          # ~2.7 hours of 1-second readings

# Random seed for reproducibility (change this to get different data)
RANDOM_SEED = 42

# Physical constants for tank simulation
TANK_1_MIN = 250                # mm — low-level alarm
TANK_1_MAX = 1200               # mm — high-level alarm
TANK_1_NORMAL_LOW = 400         # mm — normal operating range start
TANK_1_NORMAL_HIGH = 900        # mm — normal operating range end

# Attack windows: list of (start_second, end_second, attack_type, description)
ATTACK_WINDOWS = [
    (1500,  1800,  "valve_stuck",    "MV101 valve forced SHUT while pump runs — tank drains uncontrollably"),
    (3200,  3600,  "sensor_spoof",   "LIT101 level sensor spoofed to report normal while tank overflows"),
    (6000,  6300,  "pump_dos",       "P101 pump disabled — no inflow, level drops to dangerous low"),
    (8000,  8400,  "chemical_spike", "AIT202 chemical dosing manipulated — conductivity spikes abnormally"),
]

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# HELPER: Clamp a value between min and max
# ─────────────────────────────────────────────
def clamp(value, low, high):
    return max(low, min(high, value))


# ─────────────────────────────────────────────
# MAIN SIMULATION
# ─────────────────────────────────────────────
def simulate():
    print("🔄 Starting SWaT simulation...")

    # --- Build timestamp index (one row per second) ---
    start_time = datetime(2015, 12, 22, 10, 0, 0)   # matches real SWaT start date
    timestamps = [start_time + timedelta(seconds=i) for i in range(TOTAL_SECONDS)]

    # ── Stage 1 state variables ──────────────────────────────────────────────
    lit101 = 700.0      # tank 1 water level (mm), starts mid-range
    mv101 = 1           # valve: 1 = open (inlet water coming in)
    p101 = 1            # pump 101: 1 = running (sending water to stage 2)
    p102 = 0            # pump 102: backup pump, starts off

    # ── Stage 2 state variables ──────────────────────────────────────────────
    mv201 = 1
    fit201_base = 1.2   # L/s, baseline flow for stage 2

    # ── Stage 3 state variables ──────────────────────────────────────────────
    lit301 = 850.0
    mv301, mv302 = 1, 0
    p301, p302 = 1, 0

    # ── Storage lists — one entry per timestep ──────────────────────────────
    rows = []

    # ── Build a lookup set of which seconds are attack seconds ──────────────
    attack_second_lookup = {}   # second_index → attack_type
    for (start, end, atype, _) in ATTACK_WINDOWS:
        for s in range(start, end):
            attack_second_lookup[s] = atype

    # ────────────────────────────────────────────────────────────────────────
    # MAIN LOOP — simulate one second at a time
    # ────────────────────────────────────────────────────────────────────────
    for t in range(TOTAL_SECONDS):

        # Determine if this second is under attack
        attack_type = attack_second_lookup.get(t, None)
        label = "Attack" if attack_type else "Normal"

        # ── PHYSICS: Tank 1 level update ────────────────────────────────────
        # Inflow: valve must be open AND inlet flow available
        # Outflow: pump must be running to drain into stage 2
        inflow_rate  = 0.0
        outflow_rate = 0.0

        # Normal inflow when valve is open
        if mv101 == 1:
            inflow_rate = 1.8 + np.random.normal(0, 0.05)   # ~1.8 L/s with noise

        # Normal outflow when pump runs
        if p101 == 1:
            outflow_rate = 1.5 + np.random.normal(0, 0.05)  # ~1.5 L/s with noise

        # ── INJECT ATTACKS ───────────────────────────────────────────────────
        if attack_type == "valve_stuck":
            # Attack: valve is FORCED closed → no inflow → tank drains
            mv101 = 0
            inflow_rate = 0.0
            # Pump keeps running → level drops fast

        elif attack_type == "pump_dos":
            # Attack: pump disabled → no outflow, level rises
            p101 = 0
            outflow_rate = 0.0

        elif attack_type == "chemical_spike":
            # Attack: chemical dosing manipulated (only affects AIT202 later)
            pass  # handled in sensor generation below

        # Level changes by net flow (converted mm using tank cross-section ~0.5 m²)
        # 1 L/s ≈ 2 mm/s level change for our simplified tank cross-section
        net_mm_per_sec = (inflow_rate - outflow_rate) * 2.0
        lit101 = clamp(lit101 + net_mm_per_sec + np.random.normal(0, 0.3),
                       TANK_1_MIN - 50, TANK_1_MAX + 50)

        # Tank control logic (PLC-like): switch valve/pump based on level
        # (only when NOT under attack)
        if attack_type not in ("valve_stuck", "pump_dos"):
            if lit101 < TANK_1_NORMAL_LOW:
                mv101 = 1   # open valve to fill
                p101 = 0    # pause pump to not drain further
            elif lit101 > TANK_1_NORMAL_HIGH:
                mv101 = 0   # close valve to stop filling
                p101 = 1    # run pump to drain
            else:
                mv101 = 1
                p101 = 1

        # ── STAGE 1 SENSORS ──────────────────────────────────────────────────
        fit101 = inflow_rate + np.random.normal(0, 0.03)

        # Sensor spoof attack: report a FAKE normal level even though tank is overflowing
        lit101_reported = lit101
        if attack_type == "sensor_spoof":
            # Attacker sends a fake reading near normal — dangerous!
            lit101_reported = 700.0 + np.random.normal(0, 5)
            # Meanwhile the real level keeps rising (overflowing)
            lit101 = clamp(lit101 + 3.0, TANK_1_MIN, TANK_1_MAX + 100)

        # ── STAGE 2 SENSORS ──────────────────────────────────────────────────
        # Chemical concentrations vary smoothly under normal conditions
        ait201 = clamp(6.5  + np.random.normal(0, 0.1), 5.0, 8.5)      # Cl residual
        ait202_base = 200 + np.random.normal(0, 2)                       # conductivity μS/cm
        ait202 = ait202_base * (5.0 if attack_type == "chemical_spike" else 1.0)
        ait203 = clamp(7.0  + np.random.normal(0, 0.05), 6.0, 8.5)      # pH
        fit201 = fit201_base + np.random.normal(0, 0.04)
        if p101 == 0:
            fit201 = max(0, fit201 * 0.1)  # barely flowing if pump is off

        # ── STAGE 3 SENSORS ──────────────────────────────────────────────────
        # Tank 3 level oscillates in a normal range
        lit301 = clamp(lit301 + np.random.normal(0, 1.5), 700, 1100)
        dpit301 = clamp(15.0 + np.random.normal(0, 0.5), 10, 20)        # differential pressure
        fit301 = clamp(1.1 + np.random.normal(0, 0.04), 0.5, 2.0)

        # ── STAGE 4 SENSORS ──────────────────────────────────────────────────
        ait401 = clamp(0.02 + np.random.normal(0, 0.002), 0.0, 0.1)    # Cl residual
        ait402 = clamp(175  + np.random.normal(0, 2), 150, 210)         # conductivity
        fit401 = clamp(1.0  + np.random.normal(0, 0.03), 0.5, 1.8)
        uv401  = clamp(100  + np.random.normal(0, 1), 80, 120)          # UV intensity

        p401 = 1 if np.random.random() > 0.05 else 0    # rarely off
        p402, p403, p404 = 0, 0, 0

        # ── STAGE 5 SENSORS ──────────────────────────────────────────────────
        ait501 = clamp(0.01 + np.random.normal(0, 0.001), 0.0, 0.05)
        ait502 = clamp(0.01 + np.random.normal(0, 0.001), 0.0, 0.05)
        ait503 = clamp(0.01 + np.random.normal(0, 0.001), 0.0, 0.05)
        fit501 = clamp(0.9  + np.random.normal(0, 0.03), 0.5, 1.5)
        fit502 = clamp(0.4  + np.random.normal(0, 0.02), 0.2, 0.8)
        fit503 = clamp(0.4  + np.random.normal(0, 0.02), 0.2, 0.8)
        fit504 = clamp(0.8  + np.random.normal(0, 0.03), 0.5, 1.3)
        pit501 = clamp(300  + np.random.normal(0, 3), 250, 380)
        pit502 = clamp(300  + np.random.normal(0, 3), 250, 380)
        pit503 = clamp(300  + np.random.normal(0, 3), 250, 380)
        p501 = 1; p502 = 0

        # ── STAGE 6 SENSORS ──────────────────────────────────────────────────
        fit601 = clamp(0.9  + np.random.normal(0, 0.03), 0.5, 1.5)
        p601, p602, p603 = 1, 0, 0

        # ── ASSEMBLE ROW ─────────────────────────────────────────────────────
        rows.append({
            "Timestamp": timestamps[t].strftime("%d/%m/%Y %H:%M:%S"),
            # Stage 1
            "FIT101": round(fit101, 4),
            "LIT101": round(lit101_reported, 2),    # may be spoofed
            "MV101":  mv101,
            "P101":   p101,
            "P102":   p102,
            # Stage 2
            "AIT201": round(ait201, 3),
            "AIT202": round(ait202, 2),
            "AIT203": round(ait203, 3),
            "FIT201": round(fit201, 4),
            "MV201":  mv201,
            # Stage 3
            "LIT301":  round(lit301, 2),
            "DPIT301": round(dpit301, 3),
            "FIT301":  round(fit301, 4),
            "MV301":   mv301,
            "MV302":   mv302,
            "P301":    p301,
            "P302":    p302,
            # Stage 4
            "AIT401": round(ait401, 4),
            "AIT402": round(ait402, 2),
            "FIT401": round(fit401, 4),
            "P401":   p401,
            "P402":   p402,
            "P403":   p403,
            "P404":   p404,
            "UV401":  round(uv401, 2),
            # Stage 5
            "AIT501": round(ait501, 4),
            "AIT502": round(ait502, 4),
            "AIT503": round(ait503, 4),
            "FIT501": round(fit501, 4),
            "FIT502": round(fit502, 4),
            "FIT503": round(fit503, 4),
            "FIT504": round(fit504, 4),
            "P501":   p501,
            "P502":   p502,
            "PIT501": round(pit501, 2),
            "PIT502": round(pit502, 2),
            "PIT503": round(pit503, 2),
            # Stage 6
            "FIT601": round(fit601, 4),
            "P601":   p601,
            "P602":   p602,
            "P603":   p603,
            # Label
            "Normal/Attack": label,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = simulate()

    # Save to CSV in the data/ folder
    out_path = os.path.join("data", "swat_data.csv")
    df.to_csv(out_path, index=False)

    # Print a summary so you can verify it worked
    print(f"\n✅ Dataset saved to: {out_path}")
    print(f"   Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Date range   : {df['Timestamp'].iloc[0]}  →  {df['Timestamp'].iloc[-1]}")
    print(f"\n   Label distribution:")
    print(df["Normal/Attack"].value_counts().to_string())

    print("\n   Attack windows injected:")
    for start, end, atype, desc in ATTACK_WINDOWS:
        print(f"   [{start:>5}s – {end:>5}s]  {atype:20s}  {desc}")
