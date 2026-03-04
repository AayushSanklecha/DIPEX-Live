"""
colab/train_drift_autoencoder.py
----------------------------------
DIPEX — Google Colab Training Script: Autoencoder for Multivariate Drift Detection
Run this on Google Colab (free tier CPU is sufficient; GPU optional).

Outputs
-------
  models/drift_autoencoder.pkl  — MLPRegressor autoencoder
  models/drift_scaler.pkl       — Fitted StandardScaler

Instructions
------------
1. Upload your baseline dataset (CSV) to Colab or Google Drive.
2. Set BASELINE_CSV to the file path.
3. Run all cells. Training takes 2-5 minutes on Colab CPU.
4. Download the two .pkl files → place in your project's models/ folder.

If no baseline CSV is provided, a synthetic dataset is generated.
"""

# !pip install scikit-learn pandas numpy joblib -q

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ─── Config ───────────────────────────────────────────────────────────────────

BASELINE_CSV     = None    # e.g., "/content/drive/MyDrive/baseline_data.csv"
TARGET_COLUMN    = None    # Set to your target column name if you want to exclude it
N_SYNTHETIC_ROWS = 5_000   # Used only if BASELINE_CSV is None

print("=== DIPEX Drift Autoencoder Training ===")

# ─── Step 1: Load or generate baseline data ───────────────────────────────────

if BASELINE_CSV and os.path.exists(BASELINE_CSV):
    df = pd.read_csv(BASELINE_CSV)
    print(f"Loaded baseline: {df.shape}")
else:
    print(f"Generating synthetic baseline ({N_SYNTHETIC_ROWS} rows)...")
    rng = np.random.default_rng(42)
    df  = pd.DataFrame(rng.standard_normal((N_SYNTHETIC_ROWS, 10)),
                       columns=[f"feat_{i}" for i in range(10)])
    df["amount"]  = rng.exponential(scale=1000, size=N_SYNTHETIC_ROWS)
    df["age"]     = rng.integers(18, 90, size=N_SYNTHETIC_ROWS).astype(float)
    df["score"]   = rng.uniform(0, 1, size=N_SYNTHETIC_ROWS)
    print(f"Synthetic data shape: {df.shape}")

# ─── Step 2: Prepare features ─────────────────────────────────────────────────

num_cols = df.select_dtypes(include="number").columns.tolist()
if TARGET_COLUMN and TARGET_COLUMN in num_cols:
    num_cols = [c for c in num_cols if c != TARGET_COLUMN]

print(f"Using {len(num_cols)} numeric features: {num_cols}")

X = df[num_cols].fillna(df[num_cols].median()).values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, _, _ = train_test_split(X_scaled, X_scaled, test_size=0.1, random_state=42)

# ─── Step 3: Build and train autoencoder ──────────────────────────────────────

n_feat = X_scaled.shape[1]
h      = (max(n_feat * 2, 8), max(n_feat, 4), max(n_feat * 2, 8))  # bottleneck arch

print(f"\nAutoencoder architecture: {n_feat} → {h[0]} → {h[1]} → {h[2]} → {n_feat}")
print("Training...")

ae = MLPRegressor(
    hidden_layer_sizes=h,
    activation="relu",
    solver="adam",
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=42,
    verbose=True,
)
ae.fit(X_train, X_train)  # Input = Output (autoencoder)

# ─── Step 4: Compute reconstruction error on validation set ───────────────────

val_pred = ae.predict(X_val)
val_err  = np.mean(np.square(X_val - val_pred), axis=1)
p95      = np.percentile(val_err, 95)
print(f"\nValidation reconstruction error:")
print(f"  Mean: {val_err.mean():.6f}")
print(f"  Std:  {val_err.std():.6f}")
print(f"  P95:  {p95:.6f}  ← drift threshold at runtime")

# ─── Step 5: Save artifacts ───────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
joblib.dump(ae,     "models/drift_autoencoder.pkl")
joblib.dump(scaler, "models/drift_scaler.pkl")
print("\nArtifacts saved:")
print("  models/drift_autoencoder.pkl")
print("  models/drift_scaler.pkl")
print("\n>> Download both files → place in your project's models/ directory.")
print(">> DriftDetector will auto-load them next pipeline run.")
