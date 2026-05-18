#!/usr/bin/env python3
"""
================================================================
 ADAP v7 — MODEL 1/6: Drift Autoencoder  (PyTorch)
================================================================
HOW TO RUN IN COLAB:
  1. Run Cell 0 (pip install)
  2. Paste & run 00_shared_utils.py as Cell 1
  3. Paste & run THIS file as Cell 2

PyTorch is pre-installed in Colab — no extra install needed.

Architecture:
  Symmetric bottleneck AE:  input → h1 → h2(bottleneck) → h1 → input
  Regularization: BatchNorm1d + GELU + Dropout
  Optimizer: AdamW + ReduceLROnPlateau, gradient clipping
  Anomaly score: 0.6 * reconstruction_error + 0.4 * latent_dist_from_centroid
  Threshold: MAD-based (robust vs mean+3σ for skewed distributions)

OUTPUT: /content/adap_models/drift_pipeline.pkl
================================================================
"""
# ── ASSUMES 00_shared_utils.py has already been executed ──────────────────────

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Architecture ──────────────────────────────────────────────────────────────
class _DriftAE(nn.Module):
    """
    Symmetric bottleneck autoencoder with BatchNorm + GELU + Dropout.
    Linear bottleneck (no activation) is more stable for reconstruction tasks
    than a ReLU bottleneck which collapses negative latent directions.
    """
    def __init__(self, input_dim: int, h1: int, h2: int, dropout: float) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),          # linear bottleneck — no activation
        )
        self.dec = nn.Sequential(
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, input_dim),   # linear output — regression
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def _ae_fit(
    X_tr: np.ndarray, X_val: np.ndarray,
    h1: int, h2: int, lr: float, batch_size: int,
    dropout: float, weight_decay: float,
    max_epochs: int = 300, patience: int = 30,
) -> tuple:
    """Train autoencoder. Returns (model, best_val_mse)."""
    device = _TORCH_DEVICE
    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    Xva_t = torch.tensor(X_val, dtype=torch.float32)

    model = _DriftAE(X_tr.shape[1], h1, h2, dropout).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=12, factor=0.5, min_lr=1e-7)

    bs  = min(batch_size, len(X_tr))
    loader = DataLoader(
        TensorDataset(Xtr_t), batch_size=bs,
        shuffle=True, drop_last=(len(X_tr) > bs),
    )

    best_val  = float("inf")
    best_wts  = None
    no_improve = 0

    for _ in range(max_epochs):
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_mse = float(nn.functional.mse_loss(
                model(Xva_t.to(device)), Xva_t.to(device)).cpu())
        sched.step(val_mse)

        if val_mse < best_val - 1e-8:
            best_val   = val_mse
            best_wts   = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_wts:
        model.load_state_dict({k: v.to(device) for k, v in best_wts.items()})
    return model, best_val


def _ae_anomaly_scores(
    model: _DriftAE,
    X: np.ndarray,
    centroid: np.ndarray,
    recon_scale: float,
    lat_scale: float,
) -> tuple:
    """
    Dual anomaly score: 0.6 * reconstruction_MSE + 0.4 * latent_distance.
    Both components are normalized by their 95th-percentile on the training set
    so neither dominates.
    """
    device = _TORCH_DEVICE
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        recon  = model(Xt).cpu().numpy()
        latent = model.encode(Xt).cpu().numpy()
    recon_err = np.mean((X - recon) ** 2, axis=1)
    lat_dist  = np.mean((latent - centroid) ** 2, axis=1)
    scores = (0.6 * recon_err / (recon_scale + 1e-9)
              + 0.4 * lat_dist  / (lat_scale  + 1e-9))
    return scores, latent


# ── Corpus builder ─────────────────────────────────────────────────────────────
def _build_drift_corpus(all_dfs, rng):
    raw_blocks = []
    for df in all_dfs:
        num_cols = df.select_dtypes(include="number").columns
        for col in num_cols:
            s = df[col].dropna()
            if len(s) < 15:
                continue
            fp = _extract_drift_fingerprint(df[col])
            if fp is not None and np.isfinite(fp).all():
                raw_blocks.append(fp)
            # Variant 1: inject additional nulls — simulates high-missingness drift
            s_null = df[col].copy()
            null_idx = rng.choice(len(s_null), max(1, int(len(s_null) * 0.15)), replace=False)
            s_null.iloc[null_idx] = np.nan
            fp2 = _extract_drift_fingerprint(s_null)
            if fp2 is not None and np.isfinite(fp2).all():
                raw_blocks.append(fp2)
            # Variant 2: scale shift — simulates unit/currency drift
            s_shifted = df[col] * float(rng.choice([0.1, 0.5, 2.0, 10.0]))
            fp3 = _extract_drift_fingerprint(s_shifted)
            if fp3 is not None and np.isfinite(fp3).all():
                raw_blocks.append(fp3)

    if not raw_blocks:
        raise RuntimeError("Drift corpus is empty — no valid numeric columns found.")
    corpus = np.vstack(raw_blocks)
    rng.shuffle(corpus)
    return np.nan_to_num(corpus, nan=0.0, posinf=1e4, neginf=-1e4)


# ── Main training function ─────────────────────────────────────────────────────
def train_drift_autoencoder(all_dfs):
    log.info("\n=== [1/6] Drift Autoencoder (PyTorch) ===")
    t0  = time.perf_counter()
    rng = _make_rng(1)

    corpus = _build_drift_corpus(all_dfs, rng)
    log.info("  Corpus: %d fingerprints × %d features | device: %s",
             *corpus.shape, _TORCH_DEVICE)
    corpus = _clip_transform(corpus, 99.5)

    sc = RobustScaler()
    corpus_s = sc.fit_transform(corpus)

    # Drop near-zero-variance features — these cause degenerate PCA/AE
    feat_stds  = corpus_s.std(axis=0)
    valid_mask = feat_stds > 1e-6
    if valid_mask.sum() < 2:
        valid_mask = np.ones(corpus_s.shape[1], dtype=bool)
    corpus_sv  = corpus_s[:, valid_mask]
    input_dim  = int(valid_mask.sum())
    log.info("  Features after variance filter: %d / %d", input_dim, corpus_s.shape[1])

    # 80/20 holdout split — AE never sees holdout during any training
    n     = len(corpus_sv)
    idx   = rng.permutation(n)
    split = int(n * 0.80)
    X_tr, X_ho = corpus_sv[idx[:split]], corpus_sv[idx[split:]]
    # Inner validation for early stopping (15% of train)
    n_val = max(20, int(0.15 * len(X_tr)))
    X_ae_tr, X_ae_val = X_tr[:-n_val], X_tr[-n_val:]
    log.info("  AE-train: %d  AE-val: %d  Holdout: %d", len(X_ae_tr), len(X_ae_val), len(X_ho))

    # ── Optuna HPO ────────────────────────────────────────────────────────────
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def ae_obj(trial):
            h1 = trial.suggest_int("h1", 24, 96)
            h2 = trial.suggest_int("h2", 4, max(4, h1 // 2))
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            bs = trial.suggest_categorical("bs", [32, 64, 128])
            dr = trial.suggest_float("dr", 0.10, 0.35)
            wd = trial.suggest_float("wd", 1e-5, 1e-2, log=True)
            _, val_mse = _ae_fit(X_ae_tr, X_ae_val, h1, h2, lr, bs, dr, wd,
                                  max_epochs=120, patience=20)
            return val_mse

        study = optuna.create_study(direction="minimize")
        study.optimize(ae_obj, n_trials=40, show_progress_bar=True)
        bp      = study.best_params
        best_h1 = bp["h1"]; best_h2 = bp["h2"]
        best_lr = bp["lr"]; best_bs = bp["bs"]
        best_dr = bp["dr"]; best_wd = bp["wd"]
        log.info("  Optuna → h1=%d h2=%d lr=%.2e bs=%d dr=%.2f wd=%.2e val_MSE=%.6f",
                 best_h1, best_h2, best_lr, best_bs, best_dr, best_wd, study.best_value)
    except Exception as e:
        log.warning("  Optuna failed (%s) — using safe defaults", e)
        best_h1, best_h2 = 48, 10
        best_lr, best_bs = 5e-4, 64
        best_dr, best_wd = 0.20, 1e-4

    # ── Final training on full 80% train set ──────────────────────────────────
    model, _ = _ae_fit(X_ae_tr, X_ae_val, best_h1, best_h2,
                        best_lr, best_bs, best_dr, best_wd,
                        max_epochs=300, patience=30)

    # ── Compute training-set anomaly baselines for threshold calibration ───────
    model.eval()
    with torch.no_grad():
        Xtr_t    = torch.tensor(X_tr, dtype=torch.float32).to(_TORCH_DEVICE)
        recon_tr = model(Xtr_t).cpu().numpy()
        lat_tr   = model.encode(Xtr_t).cpu().numpy()
    recon_err_tr = np.mean((X_tr - recon_tr) ** 2, axis=1)
    centroid     = lat_tr.mean(axis=0)
    lat_dist_tr  = np.mean((lat_tr - centroid) ** 2, axis=1)
    # Use 95th-pct of training distribution as scale (robust to outliers in train)
    recon_scale = float(np.percentile(recon_err_tr, 95)) + 1e-9
    lat_scale   = float(np.percentile(lat_dist_tr,  95)) + 1e-9

    # Training set composite scores for threshold
    scores_tr, _ = _ae_anomaly_scores(model, X_tr, centroid, recon_scale, lat_scale)

    # ── Holdout evaluation ────────────────────────────────────────────────────
    with torch.no_grad():
        Xho_t    = torch.tensor(X_ho, dtype=torch.float32).to(_TORCH_DEVICE)
        recon_ho = model(Xho_t).cpu().numpy()
    recon_err_ho  = np.mean((X_ho - recon_ho) ** 2, axis=1)
    tr_mse        = float(recon_err_tr.mean())
    ho_mse        = float(recon_err_ho.mean())
    overfit_ratio = ho_mse / (tr_mse + 1e-9)
    log.info("  Train MSE=%.6f  Hold MSE=%.6f  Ratio=%.2f", tr_mse, ho_mse, overfit_ratio)

    # ── MAD-based anomaly threshold (robust vs mean±3σ) ───────────────────────
    med         = float(np.median(scores_tr))
    mad         = float(np.median(np.abs(scores_tr - med)))
    thr_mad     = med + 3.0 * mad
    thr_pct95   = float(np.percentile(scores_tr, 95))
    threshold   = max(thr_mad, thr_pct95)    # conservative
    log.info("  Anomaly threshold  MAD-3σ=%.4f  P95=%.4f  → using=%.4f",
             thr_mad, thr_pct95, threshold)

    # ── Quality gate ──────────────────────────────────────────────────────────
    gate_spec = GATES["drift_autoencoder"]
    if overfit_ratio > gate_spec["max_overfit_ratio"]:
        log.warning("  GATE FAIL  drift_ae  ratio=%.2f > %.1f",
                    overfit_ratio, gate_spec["max_overfit_ratio"])
    elif ho_mse < gate_spec.get("min_mse", 1e-7):
        log.warning("  SUSPECT AE  holdout MSE=%.2e near-zero (data leakage?)", ho_mse)
    else:
        log.info("  GATE PASS  drift_ae  ratio=%.2f  ho_mse=%.6f", overfit_ratio, ho_mse)

    # ── Save — dict format (not sklearn Pipeline) for PyTorch state dict ───────
    ae_artifact = {
        "type":            "torch_ae_v7",
        "input_dim":       input_dim,
        "h1":              best_h1,
        "h2":              best_h2,
        "dropout":         best_dr,
        "state_dict":      {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler":          sc,                # RobustScaler (sklearn, still pickle-able)
        "valid_mask":      valid_mask,         # feature selection mask
        "latent_centroid": centroid,
        "recon_scale":     recon_scale,
        "lat_scale":       lat_scale,
        "threshold":       threshold,
        "architecture":    f"[{DRIFT_DIM}→{best_h1}→{best_h2}→{best_h1}→{DRIFT_DIM}]",
        "feat_names":      DRIFT_FEAT_NAMES,
        "version":         VERSION,
    }
    joblib.dump(ae_artifact,   f"{MODELS_DIR}/drift_pipeline.pkl")
    joblib.dump(DRIFT_FEAT_NAMES, f"{MODELS_DIR}/drift_feature_names.pkl")

    sz = Path(f"{MODELS_DIR}/drift_pipeline.pkl").stat().st_size / 1e6
    log.info("  ✓ Saved drift_pipeline.pkl (%.1f MB) | %.1f min",
             sz, (time.perf_counter() - t0) / 60)
    save_report("drift_autoencoder", {
        "type":              "torch_ae_v7",
        "corpus_rows":       n,
        "drift_feat_dim":    DRIFT_DIM,
        "input_dim_filtered": input_dim,
        "architecture":      ae_artifact["architecture"],
        "train_recon_mse":   round(tr_mse, 6),
        "holdout_recon_mse": round(ho_mse, 6),
        "overfit_ratio":     round(overfit_ratio, 3),
        "gate_passed":       overfit_ratio < gate_spec["max_overfit_ratio"],
        "threshold_mad3s":   round(thr_mad, 4),
        "threshold_p95":     round(thr_pct95, 4),
        "threshold_used":    round(threshold, 4),
        "device":            _TORCH_DEVICE,
        "best_h1":           best_h1, "best_h2": best_h2,
        "best_lr":           best_lr, "best_dropout": best_dr,
        "version":           VERSION,
        "time_s":            round(time.perf_counter() - t0, 1),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_dfs = load_all_real(max_openml=120, use_cache=True)
    if not all_dfs:
        log.error("No datasets loaded!"); import sys; sys.exit(1)
    train_drift_autoencoder(all_dfs)
