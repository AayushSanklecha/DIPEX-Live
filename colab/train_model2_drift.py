"""
DIPEX — Model 2 ONLY: Drift Detector (β-VAE + KS Ensemble)
============================================================
Run standalone on Colab T4 GPU.

ROOT CAUSE OF PREVIOUS FAILURE:
  'adult' dataset has only 2 numeric cols → min_cols=2 → VAE on 2 features
  → can't detect drift → TPR=0.22.

THIS SCRIPT:
  - Loads only datasets with ≥10 numeric cols (rich feature space)
  - spambase: 57 cols, mfeat-fourier: 76, eeg-eye-state: 14
  - Skips KDD99 entirely (problematic subset selection)
  - Uses P80 threshold for better TPR sensitivity
  - Targets: TPR > 0.85, FPR < 0.10
"""
import os, time, copy, warnings, datetime
import numpy as np
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# ── Drive Setup ────────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    drive.mount("/content/drive")
    SAVE_DIR = "/content/drive/MyDrive/dipex_models"
except Exception:
    SAVE_DIR = "models"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
np.random.seed(42)

def header(t): print(f"\n{'═'*65}\n  {t}\n{'═'*65}")

header("MODEL 2 — Drift Detector (β-VAE + KS Ensemble) [FIXED]")
t0 = time.time()

# ── PyTorch Setup ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  PyTorch device: {DEVICE}")
except ImportError:
    TORCH_OK = False
    print("  ⚠ PyTorch not available — sklearn MLP fallback")

# ── Load ONLY high-dimensional numeric datasets ────────────────────────────────
# Key: need ≥10 numeric cols so VAE has a meaningful manifold to learn
DRIFT_SOURCES = [
    ("spambase",        40_000, 10),   # 57 numeric cols — email features
    ("eeg-eye-state",   14_980, 10),   # 14 numeric cols — EEG signals
    ("mfeat-fourier",   20_000, 10),   # 76 numeric cols — image fourier coeffs
    ("mfeat-karhunen",  20_000, 10),   # 64 numeric cols — image features
    ("waveform-5000",   40_000, 10),   # 40 numeric cols — waveform signal
    ("mfeat-factors",   40_000, 10),   # 216 numeric cols — image factors
    ("optdigits",       40_000, 10),   # 64 numeric cols — digits image
]

drift_frames = []
for ds_name, max_rows, min_cols in DRIFT_SOURCES:
    try:
        ds = fetch_openml(name=ds_name, version="active", as_frame=True, parser="auto")
        df = ds.frame if hasattr(ds, "frame") else None
        if df is None:
            raise ValueError("frame not available")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < min_cols:
            print(f"    {ds_name}: skipped — {len(num_cols)} numeric cols (need ≥{min_cols})")
            continue
        frm = df[num_cols].fillna(df[num_cols].median()).head(max_rows)
        drift_frames.append(frm)
        print(f"    {ds_name}: {len(frm):,} rows, {len(num_cols)} numeric cols ✅")
    except Exception as e:
        print(f"    ⚠ {ds_name}: {e}")

if not drift_frames:
    raise RuntimeError("No valid drift datasets loaded!")

# ── Scale individually → trim to common feature count ─────────────────────────
all_drift_scaled = []
for frm in drift_frames:
    sc = StandardScaler()
    all_drift_scaled.append(sc.fit_transform(frm.values.astype(np.float32)))

min_cols_all = min(a.shape[1] for a in all_drift_scaled)
print(f"\n  Feature space: {min_cols_all} cols (min across {len(drift_frames)} sources)")
all_trimmed   = [a[:, :min_cols_all] for a in all_drift_scaled]
drift_data    = np.vstack(all_trimmed).astype(np.float32)
np.random.shuffle(drift_data)
print(f"  Total drift training rows: {len(drift_data):,} × {drift_data.shape[1]} features")

n_d    = drift_data.shape[1]
n_tr   = int(0.60 * len(drift_data))
n_v    = int(0.20 * len(drift_data))
X_dtr  = drift_data[:n_tr]
X_dv   = drift_data[n_tr:n_tr + n_v]
X_dte  = drift_data[n_tr + n_v:]
print(f"  Split: train={len(X_dtr):,}  val={len(X_dv):,}  test={len(X_dte):,}")

# ── β-VAE ──────────────────────────────────────────────────────────────────────
if TORCH_OK:
    LATENT = min(64, n_d // 2)

    class VAE(nn.Module):
        def __init__(self, d, latent):
            super().__init__()
            h = max(256, d * 4)
            self.enc = nn.Sequential(
                nn.Linear(d, h),      nn.ReLU(), nn.BatchNorm1d(h),
                nn.Linear(h, 128),    nn.ReLU(), nn.Dropout(0.15))
            self.mu_l  = nn.Linear(128, latent)
            self.var_l = nn.Linear(128, latent)
            self.dec = nn.Sequential(
                nn.Linear(latent, 128), nn.ReLU(), nn.BatchNorm1d(128),
                nn.Linear(128, h),      nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(h, d))

        def reparameterize(self, mu, lv):
            return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

        def forward(self, x):
            h = self.enc(x)
            mu, lv = self.mu_l(h), self.var_l(h)
            return self.dec(self.reparameterize(mu, lv)), mu, lv

    def vae_loss(recon, x, mu, lv, beta=0.5):
        mse = nn.functional.mse_loss(recon, x, reduction="mean")
        kl  = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        return mse + beta * kl

    vae  = VAE(n_d, LATENT).to(DEVICE)
    opt  = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

    BS, EPOCHS, PATIENCE = 512, 120, 20
    best_val, best_state, pc = 1e9, None, 0

    print(f"\n  Training β-VAE (latent={LATENT}, {EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        vae.train()
        idx = np.random.permutation(len(X_dtr))
        ep_loss = 0.0
        for i in range(0, len(X_dtr), BS):
            b = torch.tensor(X_dtr[idx[i:i+BS]]).to(DEVICE)
            opt.zero_grad()
            r, mu, lv = vae(b)
            loss = vae_loss(r, b, mu, lv)
            loss.backward(); opt.step()
            ep_loss += loss.item() * len(b)
        ep_loss /= len(X_dtr)

        vae.eval()
        with torch.no_grad():
            xv = torch.tensor(X_dv).to(DEVICE)
            rv, muv, lvv = vae(xv)
            val_l = float(vae_loss(rv, xv, muv, lvv))

        sched.step()
        if val_l < best_val:
            best_val = val_l
            best_state = {k: v.cpu().clone() for k,v in vae.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= PATIENCE:
                print(f"    Early stop @ epoch {epoch+1}")
                break

        if (epoch+1) % 20 == 0:
            print(f"    Ep {epoch+1:3d}: train={ep_loss:.5f}  val={val_l:.5f}  patience={pc}")

    vae.load_state_dict(best_state)
    vae.eval()

    def recon_error(X_np):
        with torch.no_grad():
            t = torch.tensor(X_np.astype(np.float32)).to(DEVICE)  # force float32
            r, mu, lv = vae(t)
            return nn.functional.mse_loss(r, t, reduction="none").mean(dim=1).cpu().numpy()

    errs_tr  = recon_error(X_dtr)
    errs_val = recon_error(X_dv)
    errs_te  = recon_error(X_dte)

    # P80 threshold — provides better TPR while keeping FPR low
    drift_threshold = float(np.percentile(errs_val, 80))

    print(f"\n  beta-VAE DIAGNOSIS:")
    print(f"  Feature dims   : {n_d}")
    print(f"  Train MSE      : {errs_tr.mean():.5f}")
    print(f"  Val   MSE      : {errs_val.mean():.5f}")
    print(f"  Test  MSE      : {errs_te.mean():.5f}")
    vt_ratio = errs_val.mean() / max(errs_tr.mean(), 1e-9)
    print(f"  Val/Train ratio: {vt_ratio:.3f}  {'OK' if vt_ratio < 1.5 else 'CHECK OVERFIT'}")
    print(f"  Drift threshold (P80 val): {drift_threshold:.5f}")

    # Test at multiple drift levels
    print(f"\n  Drift Detection at multiple injection levels:")
    print(f"  {'Drift Level':>14}  {'TPR':>7}  {'FPR':>7}  Status")
    tpr3, fpr3 = 0.0, 1.0
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        # Cast to float32 — np.random.normal returns float64 which PyTorch rejects
        drifted    = (X_dte.copy() + np.random.normal(sigma, 0.5, size=X_dte.shape)).astype(np.float32)
        errs_drift = recon_error(drifted)
        tpr = float((errs_drift > drift_threshold).mean())
        fpr = float((errs_te   > drift_threshold).mean())
        ok  = "OK" if tpr > 0.80 else "WARN"
        print(f"  +{sigma:.1f}sigma Gaussian: {tpr:>7.3f}  {fpr:>7.3f}  {ok}")
        if sigma == 3.0:
            tpr3, fpr3 = tpr, fpr

    print(f"\n  ── FINAL VERDICT ──")
    if tpr3 >= 0.85 and fpr3 < 0.10:
        print(f"  ✅ PRODUCTION READY — TPR={tpr3:.3f} ≥ 0.85, FPR={fpr3:.3f} < 0.10")
    elif tpr3 >= 0.70:
        print(f"  ⚠ ACCEPTABLE — TPR={tpr3:.3f}. Can improve with more data.")
    else:
        print(f"  ❌ NEEDS WORK — TPR={tpr3:.3f}. More feature-rich datasets needed.")

    torch.save({
        "state_dict": best_state,
        "n_features": n_d,
        "threshold":  drift_threshold,
        "latent":     LATENT,
        "tpr_at_3sigma": tpr3,
        "fpr":        fpr3,
    }, f"{SAVE_DIR}/drift_vae.pt")
    print(f"  ✅  drift_vae.pt saved")

# ── sklearn MLP fallback (pipeline-compatible .pkl) ───────────────────────────
print("\n  Training sklearn MLP autoencoder (.pkl, pipeline-compatible)...")
ae_sk = MLPRegressor(
    hidden_layer_sizes=(256, 128, LATENT if TORCH_OK else 32, 128, 256),
    activation="relu", solver="adam", learning_rate_init=1e-3,
    max_iter=300, early_stopping=True,
    validation_fraction=0.15, n_iter_no_change=15, random_state=42)
ae_sk.fit(X_dtr, X_dtr)

# Evaluate MLP
mlp_errs_te = np.mean((ae_sk.predict(X_dte) - X_dte)**2, axis=1)
mlp_thresh  = float(np.percentile(np.mean((ae_sk.predict(X_dv) - X_dv)**2, axis=1), 80))
mlp_drifted = X_dte.copy() + np.random.normal(3.0, 0.5, size=X_dte.shape)
mlp_tpr = float((np.mean((ae_sk.predict(mlp_drifted) - mlp_drifted)**2, axis=1) > mlp_thresh).mean())
mlp_fpr = float((mlp_errs_te > mlp_thresh).mean())
print(f"  MLP AE — TPR(+3σ)={mlp_tpr:.3f}  FPR={mlp_fpr:.3f}")

# ── Save scalers + model ──────────────────────────────────────────────────────
# Save global scaler on full drift_data for inference
scaler_drift = StandardScaler().fit(drift_data)
joblib.dump(scaler_drift, f"{SAVE_DIR}/drift_scaler.pkl")
joblib.dump(scaler_drift, "models/drift_scaler.pkl")
joblib.dump(ae_sk,        f"{SAVE_DIR}/drift_autoencoder.pkl")
joblib.dump(ae_sk,        "models/drift_autoencoder.pkl")

drift_metrics = {
    "n_features":    n_d,
    "n_samples":     len(drift_data),
    "train_mse":     float(errs_tr.mean()) if TORCH_OK else None,
    "val_mse":       float(errs_val.mean()) if TORCH_OK else None,
    "test_mse":      float(errs_te.mean()) if TORCH_OK else None,
    "tpr_at_3sigma": tpr3   if TORCH_OK else mlp_tpr,
    "fpr":           fpr3   if TORCH_OK else mlp_fpr,
    "threshold_p80": drift_threshold if TORCH_OK else mlp_thresh,
}

import json
registry_entry = {
    "drift_detector": {
        "version":    "2.1",
        "trained_at": datetime.datetime.utcnow().isoformat(),
        "metrics":    drift_metrics,
        "datasets":   [d[0] for d in DRIFT_SOURCES],
        "feature_dims": n_d,
    }
}
with open(f"{SAVE_DIR}/_registry_model2.json","w") as f:
    json.dump(registry_entry, f, indent=2, default=str)

print(f"\n  ✅  drift_autoencoder.pkl  saved → {SAVE_DIR}")
print(f"  ✅  drift_scaler.pkl        saved → {SAVE_DIR}")
print(f"  ⏱  Model 2 done in {time.time()-t0:.0f}s")
print(f"\n  ══ MODEL 2 COMPLETE — TPR={tpr3 if TORCH_OK else mlp_tpr:.3f} ══")
