#!/usr/bin/env python3
"""
================================================================
 ADAP v7 — MODEL 6/6: Proposal Confidence Scorer
================================================================
HOW TO RUN IN COLAB:
  1. Run Cell 0 (pip install)
  2. Paste & run 00_shared_utils.py as Cell 1
  3. Paste & run THIS file as Cell 7

OUTPUT: /content/adap_models/proposal_confidence.pkl
Expected time: ~20-25 minutes (60 Optuna trials)
================================================================
"""

CONF_FEATS = [
    "null_rate","anomaly_rate","drift_psi","data_health",
    "n_regulatory_checked","rules_passed_ratio","rules_warned_ratio","rules_failed_ratio",
    "model_auc","cv_std","quarantine_frac","retry_count","pipeline_success",
    "n_features","log_n_rows","has_target","schema_complexity","domain_enc",
    "n_missing_cols","pct_numeric","pct_categorical",
    "null_rate_sq","auc_sq","health_x_auc",
]
N_CONF = len(CONF_FEATS)

# Monotonicity constraints: +1=higher→higher conf, -1=higher→lower conf, 0=free
CONF_MONO = {
    "null_rate": -1, "anomaly_rate": -1, "drift_psi": -1,
    "data_health": +1, "n_regulatory_checked": 0,
    "rules_passed_ratio": +1, "rules_warned_ratio": 0, "rules_failed_ratio": -1,
    "model_auc": +1, "cv_std": -1, "quarantine_frac": -1, "retry_count": -1,
    "pipeline_success": +1, "n_features": 0, "log_n_rows": 0,
    "has_target": +1, "schema_complexity": 0, "domain_enc": 0,
    "n_missing_cols": -1, "pct_numeric": 0, "pct_categorical": 0,
    "null_rate_sq": -1, "auc_sq": +1, "health_x_auc": +1,
}
MONOTONE_CONSTRAINTS = [CONF_MONO.get(f, 0) for f in CONF_FEATS]


def _conf_sample_real(rng, ref_dfs):
    """
    Grounded confidence samples using REAL dataset statistics as anchors.
    Uses actual null rates, n_rows from real data — not pure uniform sampling.
    """
    df = ref_dfs[int(rng.integers(0, len(ref_dfs)))]
    num_cols    = df.select_dtypes(include="number").columns
    actual_null = float(df.isnull().mean().mean())
    actual_ncols= float(df.shape[1])
    actual_nrows= float(len(df))
    actual_pct  = len(num_cols) / max(df.shape[1], 1)

    null_rate  = float(np.clip(actual_null + rng.normal(0, 0.05), 0, 0.9))
    n_features = float(np.clip(actual_ncols + rng.integers(-5, 5), 2, 300))
    log_n_rows = float(np.log10(max(actual_nrows * rng.uniform(0.5, 2.0), 10)))
    pct_num    = float(np.clip(actual_pct + rng.normal(0, 0.1), 0, 1))

    f = {
        "null_rate":            null_rate,
        "anomaly_rate":         float(rng.uniform(0.0, 0.25)),
        "drift_psi":            float(rng.uniform(0.0, 0.8)),
        "data_health":          float(np.clip(80 - null_rate*100 - rng.uniform(0,20), 10, 100)),
        "n_regulatory_checked": float(rng.integers(0, 18)),
        "rules_passed_ratio":   float(rng.uniform(0.0, 1.0)),
        "rules_warned_ratio":   float(rng.uniform(0.0, 0.4)),
        "rules_failed_ratio":   float(rng.uniform(0.0, 0.35)),
        "model_auc":            float(rng.uniform(0.5, 1.0)),
        "cv_std":               float(rng.uniform(0.0, 0.18)),
        "quarantine_frac":      float(rng.uniform(0.0, 0.4)),
        "retry_count":          float(rng.integers(0, 4)),
        "pipeline_success":     float(rng.random() > 0.1),
        "n_features":           n_features,
        "log_n_rows":           log_n_rows,
        "has_target":           float(rng.random() > 0.3),
        "schema_complexity":    float(rng.uniform(0.1, 1.0)),
        "domain_enc":           float(rng.integers(0, 7)),
        "n_missing_cols":       float(rng.integers(0, max(int(n_features*0.3), 1))),
        "pct_numeric":          pct_num,
        "pct_categorical":      float(np.clip(1.0 - pct_num + rng.normal(0, 0.1), 0, 1)),
    }
    f["null_rate_sq"] = f["null_rate"] ** 2
    f["auc_sq"]       = (f["model_auc"] - 0.5) ** 2
    f["health_x_auc"] = (f["data_health"] / 100) * f["model_auc"]

    conf_score = (
        0.30 * f["data_health"] / 100
        + 0.25 * max(f["model_auc"] - 0.5, 0) / 0.5
        + 0.18 * f["pipeline_success"]
        + 0.10 * f["rules_passed_ratio"]
        - 0.15 * f["null_rate"]
        - 0.08 * f["anomaly_rate"]
        - 0.10 * f["quarantine_frac"]
        - 0.07 * f["rules_failed_ratio"]
        - 0.05 * f["cv_std"]
        + 0.03 * f["has_target"]
        - 0.02 * min(f["retry_count"] / 3, 1)
        - 0.05 * f["drift_psi"]
        + float(rng.normal(0, 0.03))
    )
    return f, int(conf_score > 0.58)


def train_confidence_scorer(all_dfs):
    log.info("\n=== [6/6] Confidence Scorer (monotone constraints + true 4-way split) ===")
    t0 = time.perf_counter()
    import lightgbm as lgb
    rng = _make_rng(6)

    N = 15000
    rows, ys = [], []
    ref_dfs = [d for d in all_dfs if len(d) >= 50] or all_dfs[:5]
    for _ in range(N):
        f, y = _conf_sample_real(rng, ref_dfs)
        rows.append([f.get(k, 0.0) for k in CONF_FEATS])
        ys.append(y)

    X = np.array(rows, dtype=np.float32)
    y = np.array(ys)
    log.info("  %d × %d  Class balance: %.1f%% high-conf", len(X), N_CONF, 100*y.mean())
    X = _clip_transform(X, 99.5)

    # True 4-way split: train / val / calibration / holdout  (no set reuse)
    X_base, X_hold, y_base, y_hold = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_tv, X_cal, y_tv, y_cal = train_test_split(
        X_base, y_base, test_size=0.125, stratify=y_base, random_state=SEED)  # 10% of total
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.143, stratify=y_tv, random_state=SEED)        # ~10% of total
    log.info("  Split: train=%d val=%d cal=%d hold=%d",
             len(X_tr), len(X_val), len(X_cal), len(X_hold))

    sc = RobustScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    X_cal_s = sc.transform(X_cal)
    X_ho_s  = sc.transform(X_hold)

    X_tr_b, y_tr_b = _smote_safe(X_tr_s, y_tr, SEED)

    # Optuna
    try:
        import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

        def c_obj(trial):
            # Regularisation-first search space
            p = dict(
                n_estimators       = trial.suggest_int("n",   500, 3000),
                max_depth          = trial.suggest_int("d",   3,   10),
                num_leaves         = trial.suggest_int("l",   31,  150),
                min_child_samples  = trial.suggest_int("mcs", 25,  100),
                min_split_gain     = trial.suggest_float("msg", 0.0, 0.5),
                subsample          = trial.suggest_float("ss",  0.60, 0.95),
                colsample_bytree   = trial.suggest_float("cs",  0.55, 0.95),
                reg_lambda         = trial.suggest_float("rl",  1.5,  30,  log=True),
                reg_alpha          = trial.suggest_float("ra",  0.0,  5.0),
                learning_rate      = trial.suggest_float("lr",  0.005, 0.08, log=True),
            )
            m = lgb.LGBMClassifier(
                **p, monotone_constraints=MONOTONE_CONSTRAINTS,
                random_state=SEED, n_jobs=-1, verbose=-1,
            )
            m.fit(X_tr_b, y_tr_b, eval_set=[(X_val_s, y_val)],
                  callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(-1)])
            return min(roc_auc_score(y_val, m.predict_proba(X_val_s)[:, 1]), 0.999)

        study = optuna.create_study(direction="maximize")
        study.optimize(c_obj, n_trials=60, show_progress_bar=True)
        bp = study.best_params
        best_p = dict(
            n_estimators=bp["n"], max_depth=bp["d"], num_leaves=bp["l"],
            min_child_samples=bp["mcs"], min_split_gain=bp["msg"],
            subsample=bp["ss"], colsample_bytree=bp["cs"],
            reg_lambda=bp["rl"], reg_alpha=bp["ra"], learning_rate=bp["lr"],
        )
        log.info("  Optuna val_AUC=%.4f  n=%d  leaves=%d  mcs=%d  lambda=%.2f",
                 study.best_value, bp["n"], bp["l"], bp["mcs"], bp["rl"])
    except ImportError:
        # Regularisation-first safe defaults
        best_p = dict(n_estimators=2000, max_depth=8, num_leaves=127,
                      min_child_samples=25, min_split_gain=0.1,
                      subsample=0.80, colsample_bytree=0.80,
                      reg_lambda=2.5, reg_alpha=0.3, learning_rate=0.03)

    # Final base model with monotone constraints
    base = lgb.LGBMClassifier(
        **best_p, monotone_constraints=MONOTONE_CONSTRAINTS,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    base.fit(X_tr_b, y_tr_b, eval_set=[(X_val_s, y_val)],
             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    # Platt calibration on dedicated calibration set only (never val or holdout)
    # sklearn >= 1.6: cv='prefit' is deprecated — use FrozenEstimator instead.
    log.info("  Applying Platt calibration on calibration set (n=%d)...", len(X_cal))
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    except ImportError:
        # sklearn < 1.6 fallback (FrozenEstimator not available yet)
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal_s, y_cal)

    raw_prob     = base.predict_proba(X_val_s)[:, 1]
    cal_val      = calibrated.predict_proba(X_val_s)[:, 1]
    cal_hold     = calibrated.predict_proba(X_ho_s)[:, 1]
    val_auc_raw  = roc_auc_score(y_val, raw_prob)
    val_auc_cal  = roc_auc_score(y_val, cal_val)
    hold_auc     = roc_auc_score(y_hold, cal_hold)
    ece_before   = _ece(y_val, raw_prob)
    ece_after    = _ece(y_val, cal_val)
    log.info("  Val AUC raw=%.4f → cal=%.4f  ECE %.4f→%.4f",
             val_auc_raw, val_auc_cal, ece_before, ece_after)
    log.info("  Holdout AUC=%.4f", hold_auc)

    # CV: exact same params + monotone constraints; scaler already fit on train
    cv_sc = cross_val_score(
        lgb.LGBMClassifier(**best_p, monotone_constraints=MONOTONE_CONSTRAINTS,
                           random_state=SEED, n_jobs=-1, verbose=-1),
        sc.transform(_clip_transform(X_tv, 99.5)), y_tv,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="roc_auc", n_jobs=1,
    )
    log.info("  5-Fold CV AUC: %.4f ± %.4f", cv_sc.mean(), cv_sc.std())
    log.info("  Single-split val_AUC_cal=%.4f  (informational only — gate uses CV mean)",
             val_auc_cal)
    gate = quality_gate(cv_sc.mean(), hold_auc, cv_sc.std(), "proposal_confidence")

    conf_pipeline = Pipeline([("scaler", sc), ("model", calibrated)])
    ver = _model_hash(best_p)
    joblib.dump(conf_pipeline, f"{MODELS_DIR}/proposal_confidence.pkl")
    # Metadata saved as JSON (script 07 loads with json.load)
    conf_meta = {
        "feature_names": CONF_FEATS, "n_features": N_CONF,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "val_auc_raw": round(val_auc_raw, 4), "val_auc_cal": round(val_auc_cal, 4),
        "holdout_auc_cal": round(hold_auc, 4),
        "ece_before": round(ece_before, 4), "ece_after": round(ece_after, 4),
        "cv_auc_mean": round(float(cv_sc.mean()), 4),
        "cv_auc_std": round(float(cv_sc.std()), 4),
        "quality_gate": gate, "best_params": best_p,
        "calibration_method": "sigmoid_prefit_on_calibration_set_only",
        "version": VERSION, "param_hash": ver,
    }
    with open(f"{MODELS_DIR}/confidence_metadata.json", "w") as _fj:
        import json as _json
        _json.dump(conf_meta, _fj, indent=2, default=str)

    sz = Path(f"{MODELS_DIR}/proposal_confidence.pkl").stat().st_size / 1e6
    log.info("  ✓ Saved proposal_confidence.pkl (%.1f MB)  |  %.1f min",
             sz, (time.perf_counter()-t0)/60)
    save_report("confidence_scorer", {
        "n_features": N_CONF, "n_samples_train": N,
        "val_auc_cal": round(val_auc_cal, 4), "holdout_auc_cal": round(hold_auc, 4),
        "ece_after": round(ece_after, 4), "quality_gate": gate,
        "monotone_constraints_applied": True,
        "calibration": "sigmoid_prefit_on_calibration_set_only",
        "model_size_mb": round(sz, 2), "version": VERSION,
        "time_s": round(time.perf_counter()-t0, 1),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_dfs = load_all_real(max_openml=120, use_cache=True)
    if not all_dfs:
        log.error("No datasets loaded!"); import sys; sys.exit(1)
    train_confidence_scorer(all_dfs)
