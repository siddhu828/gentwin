"""
train.py — Preprocessing, Training, and Anomaly Scoring
---------------------------------------------------------
WHAT THIS SCRIPT DOES (in order):
    1. PREPROCESS  — load CSV, select numeric sensors, standardize values
    2. SPLIT       — separate Normal rows (for training) from all rows (for testing)
    3. TRAIN       — fit the VAE on normal data only for N epochs
    4. SCORE       — compute reconstruction error for every row (normal + attack)
    5. SAVE        — write anomaly scores + labels to CSV for the dashboard

HOW TO RUN:
    python train.py

OUTPUTS:
    outputs/anomaly_scores.csv   — one row per data point, with reconstruction error
    outputs/vae_model.pt         — saved PyTorch model weights
    outputs/training_loss.png    — training curve (loss over epochs)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import our VAE class and loss function from model.py
from model import VAE, vae_loss

# ─────────────────────────────────────────────
# CONFIGURATION — tune these to experiment
# ─────────────────────────────────────────────

DATA_PATH    = "data/swat_data.csv"
OUTPUT_DIR   = "outputs"
MODEL_PATH   = os.path.join(OUTPUT_DIR, "vae_model.pt")
SCORES_PATH  = os.path.join(OUTPUT_DIR, "anomaly_scores.csv")

# Model hyperparameters
HIDDEN_DIM   = 64       # size of encoder/decoder hidden layers
LATENT_DIM   = 16       # size of the compressed bottleneck (z)
BETA         = 0.5      # KL weight — lower = emphasis on reconstruction

# Training hyperparameters
BATCH_SIZE   = 256      # how many rows to process per gradient update
EPOCHS       = 60       # how many full passes through the training data
LEARNING_RATE = 1e-3    # step size for Adam optimizer
PATIENCE     = 10       # stop early if validation loss doesn't improve

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use GPU if available, otherwise CPU
# Apple Silicon Macs can use "mps" for Metal GPU acceleration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"🖥️  Using device: {DEVICE}")


# ─────────────────────────────────────────────
# STEP 1: LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 1: Loading and Preprocessing Data")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)

# ── Select only numeric sensor columns ───────────────────────────────────────
# We exclude the label column (Normal/Attack) and Timestamp.
# Binary columns (MV101, P101, etc.) are included — pump/valve states
# matter for anomaly detection (e.g., pump off + tank filling = anomalous).

label_col = "Normal/Attack"
labels    = df[label_col].values                    # keep labels for evaluation
timestamps = df["Timestamp"].values

feature_cols = [c for c in df.columns
                if c not in [label_col, "Timestamp"]]

print(f"   Feature columns selected: {len(feature_cols)}")
print(f"   Sample features: {feature_cols[:8]}...")

X = df[feature_cols].values.astype(np.float32)     # shape: (10000, 42)

# ── Standardize: zero mean, unit variance ────────────────────────────────────
# WHY? Neural networks train much better when all inputs are on the same scale.
# Without this, a sensor with range [0, 1200] dominates one with range [0, 1].
#
# StandardScaler computes: x_scaled = (x - mean) / std
# It is FIT ONLY on normal data — we don't want attack statistics
# influencing our "normal" baseline.

normal_mask  = labels == "Normal"
attack_mask  = labels == "Attack"

X_normal = X[normal_mask]     # shape: (8600, 42) — training data
X_all    = X                  # shape: (10000, 42) — scoring data (all rows)

scaler = StandardScaler()
scaler.fit(X_normal)          # learn mean/std from NORMAL data only

X_normal_scaled = scaler.transform(X_normal)    # scale normal data
X_all_scaled    = scaler.transform(X_all)       # apply same scale to everything

print(f"\n   Total rows    : {len(X):,}")
print(f"   Normal rows   : {normal_mask.sum():,}  (training set)")
print(f"   Attack rows   : {attack_mask.sum():,}  (will be scored but NOT trained on)")
print(f"\n   After scaling — Normal data stats:")
print(f"   Mean: {X_normal_scaled.mean():.4f}  (should be ~0)")
print(f"   Std : {X_normal_scaled.std():.4f}   (should be ~1)")


# ─────────────────────────────────────────────
# STEP 2: BUILD PYTORCH DATASETS & DATALOADERS
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 2: Building Datasets")
print("=" * 55)

# Convert numpy arrays to PyTorch tensors
X_normal_tensor = torch.tensor(X_normal_scaled, dtype=torch.float32)
X_all_tensor    = torch.tensor(X_all_scaled,    dtype=torch.float32)

# Split normal data into 90% train / 10% validation
# Validation lets us monitor for overfitting during training
n_total = len(X_normal_tensor)
n_train = int(0.9 * n_total)
n_val   = n_total - n_train

# Random shuffle before splitting
perm = torch.randperm(n_total)
X_train = X_normal_tensor[perm[:n_train]]
X_val   = X_normal_tensor[perm[n_train:]]

print(f"   Train set : {len(X_train):,} rows")
print(f"   Val set   : {len(X_val):,} rows")

# TensorDataset wraps a tensor so DataLoader can batch it
# DataLoader handles batching, shuffling, and efficient memory loading
train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val),   batch_size=BATCH_SIZE, shuffle=False)


# ─────────────────────────────────────────────
# STEP 3: INITIALISE THE MODEL
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 3: Initialising VAE Model")
print("=" * 55)

input_dim = X_normal_scaled.shape[1]    # number of sensor features

model = VAE(
    input_dim  = input_dim,
    hidden_dim = HIDDEN_DIM,
    latent_dim = LATENT_DIM,
).to(DEVICE)                            # move model to GPU/CPU

# Count trainable parameters (gives you a sense of model size)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Input dimension : {input_dim} features")
print(f"   Latent dimension: {LATENT_DIM}")
print(f"   Trainable params: {n_params:,}")

# Adam optimizer — adaptive learning rates per parameter
# Works well for most deep learning problems out of the box
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler: halve LR if validation loss plateaus for 5 epochs
# This helps the model "zoom in" as it gets close to the optimum
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)


# ─────────────────────────────────────────────
# STEP 4: TRAINING LOOP
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 4: Training")
print("=" * 55)
print(f"   Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  β: {BETA}")
print(f"   Early stopping patience: {PATIENCE} epochs\n")

train_losses = []    # track loss per epoch for plotting
val_losses   = []

best_val_loss   = float("inf")
patience_counter = 0

for epoch in range(1, EPOCHS + 1):

    # ── TRAINING PHASE ───────────────────────────────────────────────────────
    model.train()                       # enable dropout and batch norm training mode
    epoch_train_loss = 0.0

    for (batch_x,) in train_loader:    # DataLoader returns tuples, unpack with (x,)
        batch_x = batch_x.to(DEVICE)

        optimizer.zero_grad()           # clear gradients from previous batch

        # Forward pass
        x_recon, mu, log_var = model(batch_x)

        # Compute combined loss
        loss, recon_l, kl_l = vae_loss(batch_x, x_recon, mu, log_var, beta=BETA)

        # Backward pass — compute gradients via backpropagation
        loss.backward()

        # Clip gradients to prevent "exploding gradient" problem
        # (rare but can happen with small batches or high LR)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── VALIDATION PHASE ─────────────────────────────────────────────────────
    model.eval()                        # disable dropout, use running stats for BN
    epoch_val_loss = 0.0

    with torch.no_grad():               # don't compute gradients during validation
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(DEVICE)
            x_recon, mu, log_var = model(batch_x)
            loss, _, _ = vae_loss(batch_x, x_recon, mu, log_var, beta=BETA)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Update learning rate scheduler
    scheduler.step(avg_val_loss)

    # ── EARLY STOPPING ───────────────────────────────────────────────────────
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)  # save best model weights
        patience_counter = 0
        saved_marker = " ← saved"
    else:
        patience_counter += 1
        saved_marker = ""

    # Print progress every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        print(f"   Epoch {epoch:>3}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.5f} | "
              f"Val Loss: {avg_val_loss:.5f}{saved_marker}")

    if patience_counter >= PATIENCE:
        print(f"\n   ⏹️  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
        break

print(f"\n   ✅ Best validation loss: {best_val_loss:.5f}")
print(f"   Model saved to: {MODEL_PATH}")


# ─────────────────────────────────────────────
# STEP 5: PLOT TRAINING CURVE
# ─────────────────────────────────────────────

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(train_losses, color="#00d4ff", linewidth=1.5, label="Training Loss")
ax.plot(val_losses,   color="#ffaa00", linewidth=1.5, label="Validation Loss")
ax.set_xlabel("Epoch", color="white")
ax.set_ylabel("VAE Loss", color="white")
ax.set_title("VAE Training Curve", color="white", fontsize=12)
ax.legend(framealpha=0.3, labelcolor="white")
ax.set_facecolor("#111")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#444")
ax.spines["left"].set_color("#444")
ax.tick_params(colors="white")

plt.tight_layout()
loss_plot_path = os.path.join(OUTPUT_DIR, "training_loss.png")
plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"\n   Training curve saved → {loss_plot_path}")


# ─────────────────────────────────────────────
# STEP 6: COMPUTE ANOMALY SCORES
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 5: Computing Anomaly Scores")
print("=" * 55)

# Load best saved model weights before scoring
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Score ALL rows (normal + attack) in one pass
# We use no_grad() because we're only doing inference, not training
all_scores = []

score_loader = DataLoader(
    TensorDataset(X_all_tensor), batch_size=512, shuffle=False
)

with torch.no_grad():
    for (batch_x,) in score_loader:
        batch_x = batch_x.to(DEVICE)

        # Get reconstruction
        x_recon, mu, log_var = model(batch_x)

        # Reconstruction error per sample: mean squared error across all features
        # shape: (batch_size,) — one score per row
        # We compute MSE per row (dim=1) rather than averaging over the whole batch
        per_sample_mse = torch.mean((batch_x - x_recon) ** 2, dim=1)

        all_scores.extend(per_sample_mse.cpu().numpy())

all_scores = np.array(all_scores)

# ── Compute a threshold using training data statistics ───────────────────────
# A common heuristic: threshold = mean + k * std of NORMAL scores
# We use k=3 (3-sigma rule) — anything beyond 3 standard deviations is flagged

normal_scores = all_scores[normal_mask]
threshold     = normal_scores.mean() + 3 * normal_scores.std()
print(f"   Anomaly threshold (mean + 3σ of normal): {threshold:.6f}")

predicted_labels = np.where(all_scores > threshold, "Attack", "Normal")


# ─────────────────────────────────────────────
# STEP 7: EVALUATE PERFORMANCE
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 6: Evaluation")
print("=" * 55)

# Manual confusion matrix
# True Positive  (TP): model says Attack, truth is Attack ✅
# True Negative  (TN): model says Normal, truth is Normal ✅
# False Positive (FP): model says Attack, truth is Normal ❌ (false alarm)
# False Negative (FN): model says Normal, truth is Attack ❌ (missed attack)

TP = np.sum((predicted_labels == "Attack") & (labels == "Attack"))
TN = np.sum((predicted_labels == "Normal") & (labels == "Normal"))
FP = np.sum((predicted_labels == "Attack") & (labels == "Normal"))
FN = np.sum((predicted_labels == "Normal") & (labels == "Attack"))

precision  = TP / (TP + FP + 1e-9)
recall     = TP / (TP + FN + 1e-9)   # also called "detection rate"
f1         = 2 * precision * recall / (precision + recall + 1e-9)
accuracy   = (TP + TN) / len(labels)

print(f"   Threshold      : {threshold:.6f}")
print(f"   True Positives : {TP}   (attacks correctly detected)")
print(f"   False Positives: {FP}   (false alarms)")
print(f"   False Negatives: {FN}   (missed attacks)")
print(f"   True Negatives : {TN}")
print(f"\n   Precision : {precision:.3f}  (of flagged rows, how many ARE attacks?)")
print(f"   Recall    : {recall:.3f}  (of all attacks, how many did we catch?)")
print(f"   F1 Score  : {f1:.3f}  (harmonic mean of precision and recall)")
print(f"   Accuracy  : {accuracy:.3f}")


# ─────────────────────────────────────────────
# STEP 8: PLOT ANOMALY SCORES OVER TIME
# ─────────────────────────────────────────────

print("\n   Generating anomaly score plot...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle("VAE Anomaly Scores vs Ground Truth",
             fontsize=13, color="white", fontweight="bold")
fig.patch.set_facecolor("#0a0a0a")

time_axis = np.arange(len(all_scores))

# Shade attack regions based on TRUE labels
true_attack_mask = labels == "Attack"
edges = np.diff(true_attack_mask.astype(int), prepend=0, append=0)
attack_starts = np.where(edges[:-1] == 1)[0]
attack_ends   = np.where(edges[1:]  == -1)[0]

# ── Top subplot: raw anomaly score ───────────────────────────────────────────
ax1.plot(time_axis, all_scores, color="#00d4ff", linewidth=0.6, alpha=0.8)
ax1.axhline(threshold, color="#ff6b9d", linewidth=1.2, linestyle="--", label=f"Threshold ({threshold:.4f})")

for s, e in zip(attack_starts, attack_ends):
    ax1.axvspan(s, e, color="red", alpha=0.2)

ax1.set_ylabel("Reconstruction Error (MSE)", color="white", fontsize=9)
ax1.set_title("Anomaly Score Over Time  (red bands = true attacks)", color="#aaa", fontsize=9)
ax1.legend(fontsize=8, framealpha=0.2, labelcolor="white")
ax1.set_facecolor("#111")
ax1.tick_params(colors="white", labelsize=8)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_color("#444")
ax1.spines["left"].set_color("#444")

# ── Bottom subplot: predicted vs true labels ──────────────────────────────────
# Show predicted anomalies (orange) vs true attacks (red) side by side
ax2.fill_between(time_axis, true_attack_mask.astype(float),
                 color="red", alpha=0.4, label="True Attack", step="mid")
ax2.fill_between(time_axis, (predicted_labels == "Attack").astype(float),
                 color="orange", alpha=0.5, label="Predicted Attack", step="mid")

ax2.set_ylabel("Attack Flag (0/1)", color="white", fontsize=9)
ax2.set_xlabel("Time (seconds)", color="white", fontsize=9)
ax2.set_title("True vs Predicted Attacks", color="#aaa", fontsize=9)
ax2.legend(fontsize=8, framealpha=0.2, labelcolor="white")
ax2.set_facecolor("#111")
ax2.tick_params(colors="white", labelsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_color("#444")
ax2.spines["left"].set_color("#444")

plt.tight_layout()
score_plot_path = os.path.join(OUTPUT_DIR, "anomaly_scores_plot.png")
plt.savefig(score_plot_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"   Anomaly score plot saved → {score_plot_path}")


# ─────────────────────────────────────────────
# STEP 9: SAVE ANOMALY SCORES TO CSV
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  STEP 7: Saving Scores to CSV")
print("=" * 55)

scores_df = pd.DataFrame({
    "Timestamp"        : df["Timestamp"].values,
    "True_Label"       : labels,
    "Anomaly_Score"    : all_scores.round(6),
    "Predicted_Label"  : predicted_labels,
    "Is_True_Attack"   : (labels == "Attack").astype(int),
    "Is_Predicted_Atk" : (predicted_labels == "Attack").astype(int),
})

# Also append the original sensor values for the dashboard
for col in feature_cols:
    scores_df[col] = df[col].values

scores_df.to_csv(SCORES_PATH, index=False)

print(f"   Scores saved to: {SCORES_PATH}")
print(f"   Columns in output: {list(scores_df.columns[:6])} + {len(feature_cols)} sensor cols")

print("\n" + "=" * 55)
print("  ✅ Part 2 Complete!")
print(f"  Check outputs/ for: training_loss.png, anomaly_scores_plot.png,")
print(f"  and anomaly_scores.csv")
print("=" * 55)
