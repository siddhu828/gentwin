# GenTwin — Generative AI + Digital Twin for ICS Cyberattack Detection

> A personal learning project combining **Variational Autoencoders** and **SimPy Digital Twins** to detect cyberattacks on an industrial water treatment plant, using the SWaT dataset.

---

## What This Project Does

Water treatment plants are critical infrastructure — and increasingly targeted by cyberattacks. This project builds two complementary detection systems:

1. **VAE Anomaly Detector** — a neural network trained only on normal sensor data. At test time, anything it struggles to reconstruct is flagged as a potential attack.
2. **Digital Twin** — a physics-based simulation of the water plant running in parallel. If the sensor readings contradict what the simulation predicts should happen, that's a red flag — even if the sensor readings look normal individually (spoofing attacks).

These are combined in a **Streamlit dashboard** showing live anomaly scores, simulation outputs, and a vulnerability summary.

---

## Project Structure

```
gentwin/
├── synthetic_swat.py   # Generate fake SWaT dataset (run first if you lack real data)
├── eda.py              # Exploratory data analysis + plots
├── model.py            # VAE architecture (encoder, reparameterization, decoder)
├── train.py            # Preprocessing, training, anomaly scoring
├── digital_twin.py     # SimPy simulation of Stage 1 (tank + valve + pump)
├── dashboard.py        # Streamlit dashboard (the main UI)
├── requirements.txt    # Python dependencies
├── data/
│   └── swat_data.csv   # Dataset (generated or real)
└── outputs/
    ├── vae_model.pt              # Saved VAE weights
    ├── anomaly_scores.csv        # Per-row anomaly scores + labels
    ├── sensor_overview.png       # EDA: time series plot
    ├── sensor_distributions.png  # EDA: Normal vs Attack distributions
    ├── correlation_heatmap.png   # EDA: sensor correlation matrix
    ├── training_loss.png         # VAE training curve
    ├── anomaly_scores_plot.png   # VAE detection results
    ├── digital_twin_none.png     # Twin: normal operation
    ├── digital_twin_valve_stuck.png  # Twin: valve attack
    └── digital_twin_sensor_spoof.png # Twin: sensor spoof attack
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the synthetic dataset (skip if you have the real SWaT CSV)
```bash
python synthetic_swat.py
```
This creates `data/swat_data.csv` with 10,000 rows, 43 columns, and 4 injected attack windows.

### 3. Explore the data
```bash
python eda.py
```
Opens 3 matplotlib plots and prints dataset statistics. Check `outputs/` for saved PNGs.

### 4. Train the VAE anomaly detector
```bash
python train.py
```
Trains for up to 60 epochs on normal data only. Saves model weights and `anomaly_scores.csv`.

### 5. Run the digital twin
```bash
# Normal operation (no attack)
python digital_twin.py --attack none

# Valve forced shut
python digital_twin.py --attack valve_stuck --attack-start 120 --attack-end 300

# Sensor reading spoofed
python digital_twin.py --attack sensor_spoof --attack-start 150 --attack-end 350
```

### 6. Launch the dashboard
```bash
streamlit run dashboard.py
```
Opens at `http://localhost:8501`. Use the sidebar to select sensors, configure attacks, and run the simulation interactively.

---

## The Dataset

This project uses the **SWaT (Secure Water Treatment)** dataset from iTrust, Singapore — or a synthetic version with the same column schema.

| Column type | Examples | What it measures |
|---|---|---|
| Level sensors (LIT) | LIT101, LIT301 | Water depth in tanks (mm) |
| Flow sensors (FIT) | FIT101, FIT201 | Water flow rate (L/s) |
| Motor valves (MV) | MV101, MV201 | Valve open/closed (0/1) |
| Pumps (P) | P101, P102 | Pump on/off (0/1) |
| Chemical sensors (AIT) | AIT201–203 | pH, conductivity, Cl residual |

To use the **real SWaT dataset**: download it from [iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/), rename the CSV to `swat_data.csv`, place it in `data/`, and ensure the `Normal/Attack` label column is present.

---

## Attack Types Simulated

| Attack | What it does | Real-world damage |
|---|---|---|
| `valve_stuck` | Inlet valve locked closed while pump runs | Tank drains dry → pump cavitation, mechanical failure |
| `sensor_spoof` | LIT101 reports fake normal level | Tank overflows silently → flooding, chemical spill |
| `pump_dos` | Pump P101 disabled | No outflow → overfill risk downstream |
| `chemical_spike` | AIT202 conductivity manipulated | Contaminated water may pass quality checks |

---

## What I Learned Building This

1. **Unsupervised anomaly detection is powerful but limited** — the VAE catches ~50% of attacks with zero attack training examples. The attacks it misses (spoofing) look statistically normal.

2. **Digital twins catch what ML misses** — a physics model doesn't trust sensors. The divergence between the twin's prediction and the sensor report is an independent detection signal.

3. **The precision/recall trade-off is everywhere** — whether tuning the VAE threshold (Part 2) or the twin's divergence alert limit (Part 3), you're always choosing between false alarms and missed detections.

4. **SimPy's process-based model maps naturally to ICS** — PLCs, pumps, sensors, and attackers are all naturally concurrent processes. SimPy makes this easy to express in Python.

5. **Sensor spoofing is the hardest problem in ICS security** — if you only monitor the sensors, a spoof is invisible. Multi-layer detection (ML + physics model) is the industry answer.

---

## Tech Stack

| Library | Used for |
|---|---|
| `pandas` | Data loading, manipulation, CSV I/O |
| `numpy` | Numerical operations, array math |
| `matplotlib` | All plots (EDA, training, twin, dashboard) |
| `torch` (PyTorch) | VAE neural network, training loop |
| `scikit-learn` | StandardScaler for feature normalization |
| `simpy` | Discrete-event simulation (digital twin) |
| `streamlit` | Interactive web dashboard |

---

## References

- iTrust SWaT Dataset: https://itrust.sutd.edu.sg/itrust-labs_datasets/
- Goh et al. (2016) — "A Dataset to Support Research in the Design of Secure Water Treatment Systems"
- Kingma & Welling (2013) — "Auto-Encoding Variational Bayes" (the original VAE paper)
- Oldsmar water treatment attack (2021): https://www.bbc.com/news/world-us-canada-56047495
