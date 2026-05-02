"""
digital_twin.py — SimPy Digital Twin of SWaT Stage 1
------------------------------------------------------
WHAT THIS MODELS:
    A single stage of a water treatment plant:

        [Inlet] → (MV101 valve) → [Tank T101] → (P101 pump) → [Stage 2]

    The PLC (Programmable Logic Controller) monitors the tank level
    every second and opens/closes the valve and pump to keep the level
    in a safe operating range (400–900 mm).

    We can inject one of two attacks at a chosen time:
        A. valve_stuck  — inlet valve forced shut while pump keeps running
        B. sensor_spoof — tank level sensor reports a false "normal" reading

HOW TO RUN:
    python digital_twin.py
    # or with a specific attack:
    python digital_twin.py --attack valve_stuck --attack-start 120 --attack-end 240
    python digital_twin.py --attack sensor_spoof --attack-start 150 --attack-end 300

OUTPUT:
    outputs/digital_twin_[attack_type].png — level vs time plot
"""

import simpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# These match the synthetic SWaT parameters so the twin is consistent
# ─────────────────────────────────────────────────────────────────────────────

TANK_CAPACITY   = 1400      # mm — physical maximum before overflow
TANK_EMPTY      = 0         # mm
LEVEL_LOW_ALARM = 300       # mm — PLC opens valve if level drops here
LEVEL_HIGH_ALARM= 1000      # mm — PLC closes valve if level hits here
LEVEL_SETPOINT_LOW  = 400   # mm — normal low operating level
LEVEL_SETPOINT_HIGH = 900   # mm — normal high operating level

INFLOW_RATE  = 1.8 * 2.0   # mm/s when valve open (1.8 L/s × 2 mm per L/s)
OUTFLOW_RATE = 1.5 * 2.0   # mm/s when pump runs  (1.5 L/s × 2 mm per L/s)
NOISE_STD    = 0.3          # mm — sensor measurement noise

SIM_STEP     = 1.0          # simulation step size in seconds (1 = real-time fidelity)
SIM_DURATION = 600          # total simulation seconds (10 minutes)

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES — we log everything for plotting
# ─────────────────────────────────────────────────────────────────────────────

class SimLog:
    """Stores time-series data collected during simulation for plotting."""
    def __init__(self):
        self.times          = []    # simulation time (seconds)
        self.true_level     = []    # actual water level in tank (mm)
        self.reported_level = []    # what the sensor reports (may be spoofed)
        self.twin_level     = []    # digital twin's independent estimate
        self.valve_open     = []    # 1=open, 0=closed
        self.pump_running   = []    # 1=running, 0=off
        self.attack_active  = []    # 1=attack in progress, 0=normal

    def record(self, t, true_lv, reported_lv, twin_lv, valve, pump, attack):
        self.times.append(t)
        self.true_level.append(true_lv)
        self.reported_level.append(reported_lv)
        self.twin_level.append(twin_lv)
        self.valve_open.append(valve)
        self.pump_running.append(pump)
        self.attack_active.append(attack)


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class Valve:
    """
    Motor valve MV101 — controls inlet water flow into the tank.
    In a real plant this is an electrically-actuated ball valve.
    """
    def __init__(self):
        self.is_open = True          # starts open (water flowing in)
        self.stuck   = False         # attack state: valve forced to a position

    def open(self):
        if not self.stuck:           # ignore command if attacker has locked it
            self.is_open = True

    def close(self):
        if not self.stuck:
            self.is_open = False

    def force_close(self):
        """Called by the attacker — bypasses normal PLC control."""
        self.stuck   = True
        self.is_open = False

    def release(self):
        """Called when attack ends — restore normal operation."""
        self.stuck = False


class Pump:
    """
    Pump P101 — drives water from Tank T101 to Stage 2.
    """
    def __init__(self):
        self.is_running = True       # starts running

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False


class LevelSensor:
    """
    Level transmitter LIT101 — measures water depth in the tank.
    Normally returns the true level plus small random noise.
    Under a sensor_spoof attack, returns a pre-programmed fake value.
    """
    def __init__(self):
        self.spoofed       = False
        self.spoof_value   = 700.0   # the fake "normal" value we'll report

    def read(self, true_level):
        if self.spoofed:
            # Attacker has injected a false signal into the sensor bus
            return self.spoof_value + np.random.normal(0, 1.0)
        else:
            # Normal reading: true level + small Gaussian noise
            return true_level + np.random.normal(0, NOISE_STD)

    def start_spoof(self, fake_value=700.0):
        self.spoofed     = True
        self.spoof_value = fake_value

    def stop_spoof(self):
        self.spoofed = False


# ─────────────────────────────────────────────────────────────────────────────
# SIMPY PROCESSES
# Each function below is a SimPy generator (coroutine).
# "yield env.timeout(dt)" means: pause this process for dt seconds,
# let other processes run, then resume here.
# ─────────────────────────────────────────────────────────────────────────────

def physical_plant(env, state, valve, pump, sensor, log, attack_config):
    """
    PROCESS: The physical water tank.

    This is the "ground truth" — it simulates real physics:
        level(t+dt) = level(t) + inflow(t)×dt - outflow(t)×dt

    It also handles the sensor reading (which may be spoofed).

    Args:
        env           : SimPy environment (the simulation clock)
        state         : dict holding mutable simulation state
        valve/pump/sensor: physical component objects
        log           : SimLog instance to record data
        attack_config : dict with attack type, start, end times
    """
    while True:
        # Advance time by one simulation step
        yield env.timeout(SIM_STEP)

        t = env.now

        # ── Determine if attack is active right now ───────────────────────
        atk_start = attack_config.get("start", 9999)
        atk_end   = attack_config.get("end",   9999)
        is_attacking = atk_start <= t < atk_end

        # ── Physics: update real tank level ──────────────────────────────
        inflow  = INFLOW_RATE  if valve.is_open    else 0.0
        outflow = OUTFLOW_RATE if pump.is_running  else 0.0
        net     = (inflow - outflow) * SIM_STEP
        # Clamp to physical limits (tank can't go below empty or above capacity)
        state["true_level"] = max(TANK_EMPTY,
                              min(TANK_CAPACITY,
                                  state["true_level"] + net + np.random.normal(0, NOISE_STD)))

        # ── Sensor reading (may be spoofed) ───────────────────────────────
        reported = sensor.read(state["true_level"])
        state["reported_level"] = reported

        # ── Digital Twin: independent estimate using known valve/pump states ──
        # The twin knows MV101 and P101 states (from SCADA) but uses its
        # OWN physics model — it does NOT trust the sensor reading.
        # This is the key: divergence between twin_level and reported_level = anomaly
        twin_inflow  = INFLOW_RATE  if state["twin_valve_open"]   else 0.0
        twin_outflow = OUTFLOW_RATE if state["twin_pump_running"] else 0.0
        twin_net     = (twin_inflow - twin_outflow) * SIM_STEP
        state["twin_level"] = max(TANK_EMPTY,
                             min(TANK_CAPACITY,
                                 state["twin_level"] + twin_net))

        # ── Log everything ────────────────────────────────────────────────
        log.record(
            t            = t,
            true_lv      = state["true_level"],
            reported_lv  = reported,
            twin_lv      = state["twin_level"],
            valve        = int(valve.is_open),
            pump         = int(pump.is_running),
            attack       = int(is_attacking),
        )


def plc_controller(env, state, valve, pump, sensor):
    """
    PROCESS: The PLC (Programmable Logic Controller).

    Runs every second, reads the sensor, and decides whether to
    open/close the valve and start/stop the pump.

    This is the NORMAL control logic — it only acts on what the
    sensor tells it. If the sensor is spoofed, the PLC is fooled.
    """
    while True:
        yield env.timeout(SIM_STEP)

        # Read what the sensor reports (may be fake during spoof attack)
        reported_level = state["reported_level"]

        # Simple bang-bang control (PLC ladder logic equivalent):
        #   - If level is too low  → open valve, pause pump
        #   - If level is too high → close valve, run pump
        #   - Otherwise            → keep both running (normal filling + draining)
        if reported_level < LEVEL_SETPOINT_LOW:
            valve.open()
            pump.stop()
        elif reported_level > LEVEL_SETPOINT_HIGH:
            valve.close()
            pump.start()
        else:
            valve.open()
            pump.start()

        # Mirror the PLC's understanding of valve/pump state into twin's state
        # (The twin follows the PLC commands, not the real hardware state)
        state["twin_valve_open"]   = valve.is_open
        state["twin_pump_running"] = pump.is_running


def attacker(env, state, valve, pump, sensor, attack_config):
    """
    PROCESS: The adversary.

    Waits until the attack start time, then injects the chosen attack.
    At the end time, restores normal operation.

    Two attack types:
        valve_stuck   — physically lock the inlet valve closed
        sensor_spoof  — inject a fake level reading into the sensor bus
    """
    attack_type  = attack_config.get("type",  "none")
    attack_start = attack_config.get("start", 9999)
    attack_end   = attack_config.get("end",   9999)

    if attack_type == "none":
        return  # no attack — process exits immediately

    # ── Wait until attack start time ──────────────────────────────────────
    yield env.timeout(attack_start)
    print(f"\n🔴 [{env.now:.0f}s] ATTACK START: '{attack_type}'")

    # ── Inject the attack ─────────────────────────────────────────────────
    if attack_type == "valve_stuck":
        # Force the valve shut — pump continues draining, no new inflow
        valve.force_close()
        print(f"   Valve MV101 locked CLOSED. Pump P101 still running.")
        print(f"   Tank will drain uncontrollably until attack ends.")

    elif attack_type == "sensor_spoof":
        # Report a comfortable mid-range level while reality may be very different
        sensor.start_spoof(fake_value=700.0)
        print(f"   LIT101 sensor spoofed to report ~700mm (actual level continues changing).")
        print(f"   PLC thinks everything is fine. Real level will overflow.")

    # ── Wait until attack ends ────────────────────────────────────────────
    duration = attack_end - attack_start
    yield env.timeout(duration)

    print(f"\n🟢 [{env.now:.0f}s] ATTACK END: restoring normal operation")

    # ── Restore normal operation ──────────────────────────────────────────
    if attack_type == "valve_stuck":
        valve.release()
        print(f"   Valve MV101 unlocked. PLC resumes normal control.")

    elif attack_type == "sensor_spoof":
        sensor.stop_spoof()
        print(f"   Sensor LIT101 restored. PLC can now read true level.")


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(attack_type="none", attack_start=120, attack_end=300,
                   sim_duration=SIM_DURATION, initial_level=700.0):
    """
    Assemble all processes and run the simulation.

    Returns:
        log : SimLog with all recorded time-series data
    """
    print(f"\n{'='*55}")
    print(f"  GenTwin — Digital Twin Simulation")
    print(f"{'='*55}")
    print(f"  Duration      : {sim_duration}s")
    print(f"  Attack type   : {attack_type}")
    if attack_type != "none":
        print(f"  Attack window : {attack_start}s – {attack_end}s")
    print(f"  Initial level : {initial_level}mm")

    # ── Create SimPy environment (the simulation clock) ───────────────────
    env = simpy.Environment()

    # ── Instantiate physical components ───────────────────────────────────
    valve  = Valve()
    pump   = Pump()
    sensor = LevelSensor()
    log    = SimLog()

    # ── Shared mutable state dict ─────────────────────────────────────────
    # SimPy processes share state through this dict (avoids global variables)
    state = {
        "true_level"      : initial_level,
        "reported_level"  : initial_level,
        "twin_level"      : initial_level,   # twin starts at the same known level
        "twin_valve_open" : True,
        "twin_pump_running": True,
    }

    # ── Attack configuration ───────────────────────────────────────────────
    attack_config = {
        "type"  : attack_type,
        "start" : attack_start,
        "end"   : attack_end,
    }

    # ── Register all processes with the SimPy environment ─────────────────
    # env.process() schedules a generator as a concurrent process.
    # All processes start at t=0 and run "simultaneously" (interleaved by SimPy).
    env.process(physical_plant(env, state, valve, pump, sensor, log, attack_config))
    env.process(plc_controller(env, state, valve, pump, sensor))
    env.process(attacker(env, state, valve, pump, sensor, attack_config))

    # ── Run the simulation ─────────────────────────────────────────────────
    print(f"\n▶  Running simulation...")
    env.run(until=sim_duration)
    print(f"✅  Simulation complete ({len(log.times)} timesteps logged)")

    return log, attack_config


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_simulation(log, attack_config, save_path):
    """
    Visualise the simulation results.

    Panel 1: Water level — true vs reported vs twin estimate
    Panel 2: Valve and pump state (0/1) over time
    Panel 3: Divergence — how much the twin's estimate differs from the sensor
    """
    attack_type  = attack_config.get("type",  "none")
    attack_start = attack_config.get("start", 9999)
    attack_end   = attack_config.get("end",   9999)

    times          = np.array(log.times)
    true_level     = np.array(log.true_level)
    reported_level = np.array(log.reported_level)
    twin_level     = np.array(log.twin_level)
    valve_state    = np.array(log.valve_open)
    pump_state     = np.array(log.pump_running)

    # Divergence: |twin estimate − sensor report|
    # This is what a real anomaly detector would monitor in production
    divergence = np.abs(twin_level - reported_level)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    fig.suptitle(
        f"Digital Twin — Stage 1 Water Tank Simulation\n"
        f"Attack: '{attack_type}'  ({attack_start}s → {attack_end}s)",
        fontsize=13, fontweight="bold", color="white", y=1.01
    )
    fig.patch.set_facecolor("#0a0a0a")

    def shade_attack(ax):
        """Utility: shade the attack window in red on any axis."""
        if attack_type != "none":
            ax.axvspan(attack_start, attack_end, color="red", alpha=0.18,
                       zorder=0, label="Attack Window")

    # ── Panel 1: Water level ───────────────────────────────────────────────
    ax = axes[0]
    shade_attack(ax)

    ax.plot(times, true_level,     color="#00d4ff", lw=1.2, label="True Level (real physics)")
    ax.plot(times, reported_level, color="#ffaa00", lw=1.0, linestyle="--",
            label="Reported Level (sensor — may be spoofed)")
    ax.plot(times, twin_level,     color="#7fff7f", lw=1.0, linestyle=":",
            label="Twin Estimate (physics model)")

    # Mark safety limits
    ax.axhline(LEVEL_SETPOINT_LOW,  color="#ff6b9d", lw=0.8, linestyle="-.",
               alpha=0.6, label=f"Low setpoint ({LEVEL_SETPOINT_LOW}mm)")
    ax.axhline(LEVEL_SETPOINT_HIGH, color="#ff6b9d", lw=0.8, linestyle="-.",
               alpha=0.6, label=f"High setpoint ({LEVEL_SETPOINT_HIGH}mm)")
    ax.axhline(TANK_CAPACITY,       color="red",     lw=0.8, linestyle="--",
               alpha=0.5, label=f"Tank overflow ({TANK_CAPACITY}mm)")
    ax.axhline(TANK_EMPTY,          color="red",     lw=0.8, linestyle="--",
               alpha=0.5)

    ax.set_ylabel("Tank Level (mm)", color="white", fontsize=9)
    ax.set_ylim(-50, TANK_CAPACITY + 100)
    ax.legend(fontsize=7.5, framealpha=0.25, labelcolor="white",
              loc="upper right", ncol=2)
    ax.set_facecolor("#111")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color("#444")

    # ── Panel 2: Valve and pump state ─────────────────────────────────────
    ax2 = axes[1]
    shade_attack(ax2)

    ax2.step(times, valve_state,  where="post", color="#00d4ff", lw=1.2,
             label="MV101 Valve (1=open)")
    ax2.step(times, pump_state + 1.5, where="post", color="#ffaa00", lw=1.2,
             label="P101 Pump (1=running, offset +1.5 for visibility)")

    ax2.set_ylabel("State (0/1)", color="white", fontsize=9)
    ax2.set_ylim(-0.3, 3.0)
    ax2.set_yticks([0, 1, 1.5, 2.5])
    ax2.set_yticklabels(["OFF", "ON", "OFF", "ON"], color="white", fontsize=8)
    ax2.legend(fontsize=7.5, framealpha=0.25, labelcolor="white")
    ax2.set_facecolor("#111")
    ax2.tick_params(colors="white", labelsize=8)
    for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax2.spines[sp].set_color("#444")

    # ── Panel 3: Twin vs Sensor divergence ────────────────────────────────
    ax3 = axes[2]
    shade_attack(ax3)

    ax3.fill_between(times, divergence, color="#ff6b9d", alpha=0.6,
                     label="|Twin − Sensor|")
    # Detection threshold: if divergence exceeds this, raise alert
    detect_threshold = 50  # mm
    ax3.axhline(detect_threshold, color="yellow", lw=1.0, linestyle="--",
                label=f"Alert threshold ({detect_threshold}mm)")

    ax3.set_ylabel("Divergence |Twin−Sensor| (mm)", color="white", fontsize=9)
    ax3.set_xlabel("Simulation Time (seconds)", color="white", fontsize=9)
    ax3.legend(fontsize=7.5, framealpha=0.25, labelcolor="white")
    ax3.set_facecolor("#111")
    ax3.tick_params(colors="white", labelsize=8)
    for sp in ["top", "right"]: ax3.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax3.spines[sp].set_color("#444")

    # ── Summary annotations ────────────────────────────────────────────────
    if attack_type == "valve_stuck":
        annotation = (
            "⚠  Valve locked SHUT\n"
            "→ No inflow, pump drains tank\n"
            "→ Risk: pump runs dry, cavitation damage"
        )
    elif attack_type == "sensor_spoof":
        annotation = (
            "⚠  Sensor reporting ~700mm (FAKE)\n"
            "→ PLC does not intervene\n"
            "→ Risk: tank overflow, flooding"
        )
    else:
        annotation = "Normal operation — no attack"

    fig.text(0.02, 0.01, annotation, color="#ffaa00", fontsize=8,
             ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a",
                       edgecolor="#ffaa00", alpha=0.8))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"📊 Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GenTwin — Stage 1 Digital Twin")
    parser.add_argument("--attack",        default="none",
                        choices=["none", "valve_stuck", "sensor_spoof"],
                        help="Attack type to inject")
    parser.add_argument("--attack-start",  type=int, default=120,
                        help="Seconds into simulation when attack begins")
    parser.add_argument("--attack-end",    type=int, default=300,
                        help="Seconds into simulation when attack ends")
    parser.add_argument("--duration",      type=int, default=SIM_DURATION,
                        help="Total simulation duration in seconds")
    args = parser.parse_args()

    # Run simulation
    log, attack_config = run_simulation(
        attack_type   = args.attack,
        attack_start  = args.attack_start,
        attack_end    = args.attack_end,
        sim_duration  = args.duration,
    )

    # Plot and save
    save_path = os.path.join(OUTPUT_DIR, f"digital_twin_{args.attack}.png")
    plot_simulation(log, attack_config, save_path)

    # Print summary stats
    import numpy as np
    times       = np.array(log.times)
    true_level  = np.array(log.true_level)
    atk_mask    = np.array(log.attack_active).astype(bool)
    norm_mask   = ~atk_mask

    print(f"\n{'─'*50}")
    print(f"  SUMMARY STATISTICS")
    print(f"{'─'*50}")
    if norm_mask.any():
        print(f"  Normal period  — mean level : {true_level[norm_mask].mean():.1f}mm")
        print(f"                 — std level  : {true_level[norm_mask].std():.1f}mm")
    if atk_mask.any():
        print(f"  Attack period  — mean level : {true_level[atk_mask].mean():.1f}mm")
        print(f"                 — min level  : {true_level[atk_mask].min():.1f}mm")
        print(f"                 — max level  : {true_level[atk_mask].max():.1f}mm")
        divergence = np.abs(np.array(log.twin_level) - np.array(log.reported_level))
        print(f"  Max twin/sensor divergence  : {divergence[atk_mask].max():.1f}mm")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
