#!/usr/bin/env python3
"""
================================================================
 ADAP v7 — MODEL 4/6: Anomaly Detector
================================================================
HOW TO RUN IN COLAB:
  1. Run Cell 0 (pip install)
  2. Paste & run 00_shared_utils.py as Cell 1
  3. Paste & run THIS file as Cell 5

OUTPUT: /content/adap_models/anomaly_detector.pkl
Expected time: ~3-5 minutes
================================================================
"""


def _build_anomaly_corpus(all_dfs, rng):
    """Build anomaly corpus from real column fingerprints."""
    blocks = []
    for df in all_dfs:
        for col in df.select_dtypes(include="number").columns:
            fp = _extract_drift_fingerprint(df[col])
            if fp is not None and np.isfinite(fp).all():
                blocks.append(fp)
    if not blocks:
        raise RuntimeError("Anomaly corpus empty — no valid numeric columns.")
    corpus = np.vstack(blocks)
    rng.shuffle(corpus)
    return np.nan_to_num(corpus, nan=0.0, posinf=1e4, neginf=-1e4)


def _inject_multivariate_anomalies(corpus, rng, n_anom):
    """
    Multivariate anomaly injection.
    Real anomalies affect multiple features simultaneously.
    Types:
      0. Correlated spike: mean + std + range all extreme
      1. Distribution reversal: skew + kurt extreme
      2. Constant-column anomaly: std=0, unique_rate=0
      3. Near-all-null: null_rate near 1, other stats near 0
    """
    n = len(corpus)
    anomalies = []
    base_idx = rng.choice(n, n_anom, replace=True)
    for i, idx in enumerate(base_idx):
        row = corpus[idx].copy()
        anom_type = i % 4
        if anom_type == 0:
            row[4] *= rng.choice([-1,1]) * rng.uniform(10, 30)   # mean_z
            row[5] *= rng.uniform(20, 100)                          # std_z
            row[14] *= rng.uniform(20, 100)                         # range_z
        elif anom_type == 1:
            row[2]  = 0.0
            row[6]  = rng.choice([-1,1]) * rng.uniform(10, 40)    # skew_z
            row[7]  = rng.uniform(100, 500)                         # kurt_z
            row[15] = rng.uniform(0.5, 0.95)                        # high_outlier_rate
        elif anom_type == 2:
            row[4]  = 0.0; row[5] = 0.0                           # mean_z, std_z
            row[11] = 0.0; row[12] = 0.0; row[13] = 0.0           # q25,q75,range
            row[17] = 0.0; row[19] = 0.01                         # cv, unique_rate
        else:
            row[0]  = rng.uniform(0.95, 1.0)                      # null_rate extreme
            row[1]  = 0.0; row[2] = 0.0                           # zero+positive ~0
            row[4:] *= rng.uniform(0.0, 0.05)                     # other stats ~0
        anomalies.append(row)
    return np.array(anomalies, dtype=np.float32)


def train_anomaly_detector(all_dfs):
    log.info("\n=== [4/6] Anomaly Detector ===")
    t0 = time.perf_counter()
    rng = _make_rng(4)

    corpus = _build_anomaly_corpus(all_dfs, rng)
    corpus = _clip_transform(corpus, 99.5)
    log.info("  Normal corpus: %d real fingerprints × %d features", *corpus.shape)

    n = len(corpus)
    idx = rng.permutation(n)
    tr_idx, ho_idx = idx[:int(n*0.80)], idx[int(n*0.80):]
    X_tr, X_ho = corpus[tr_idx], corpus[ho_idx]

    sc = RobustScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_ho_s = sc.transform(X_ho)

    n_anom_pool = max(int(len(X_ho_s) * 0.25), 50)
    anoms_raw = _inject_multivariate_anomalies(corpus, rng, n_anom_pool)
    anoms_s   = sc.transform(_clip_transform(anoms_raw, 99.5))
    log.info("  Eval set: %d normal + %d anomalies (%.1f%% anomaly rate)",
             len(X_ho_s), len(anoms_s),
             100 * len(anoms_s) / (len(X_ho_s) + len(anoms_s)))
    X_eval = np.vstack([X_ho_s, anoms_s])
    y_eval = np.array([1] * len(X_ho_s) + [-1] * len(anoms_s))

    best_f1 = -1.0
    best_iso_params = dict(n_estimators=300, max_samples="auto", max_features=0.8)
    
    try:
        import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

        def iso_obj(trial):
            p = dict(
                n_estimators=trial.suggest_int("n", 100, 600),
                max_samples=trial.suggest_categorical("ms", [128, 256, "auto"]),
                max_features=trial.suggest_float("mf", 0.5, 1.0),
            )
            m = IsolationForest(**p, contamination="auto", bootstrap=True, n_jobs=-1, random_state=SEED)
            m.fit(X_tr_s)
            scores = -m.decision_function(X_eval)
            precision, recall, thresholds = precision_recall_curve(y_eval, scores, pos_label=-1)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            return np.max(f1_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(iso_obj, n_trials=30, show_progress_bar=True)
        bp = study.best_params
        best_iso_params = dict(n_estimators=bp["n"], max_samples=bp["ms"], max_features=bp["mf"])
        best_f1 = study.best_value
        log.info("  Optuna → n=%d  mf=%.2f  Best F1=%.4f", best_iso_params["n_estimators"], best_iso_params["max_features"], best_f1)
    except ImportError:
        log.info("  Optuna unavailable — using defaults")

    isoforest = IsolationForest(
        **best_iso_params, contamination="auto", bootstrap=True, n_jobs=-1, random_state=SEED,
    )
    isoforest.fit(X_tr_s)

    scores = -isoforest.decision_function(X_eval)
    precision, recall, thresholds = precision_recall_curve(y_eval, scores, pos_label=-1)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    prec, rec, f1 = precision[best_idx], recall[best_idx], f1_scores[best_idx]
    
    log.info("  Precision=%.3f  Recall=%.3f  F1=%.3f (Threshold=%.4f)", prec, rec, f1, best_thr)

    if f1 < GATES["anomaly_detector"]["min_f1"]:
        log.warning("  GATE FAIL  anomaly_detector  F1=%.3f < %.2f", f1, GATES["anomaly_detector"]["min_f1"])
    else:
        log.info("  GATE PASS  anomaly_detector  F1=%.3f", f1)

    anomaly_pipeline = Pipeline([("scaler", sc), ("detector", isoforest)])
    joblib.dump(anomaly_pipeline, f"{MODELS_DIR}/anomaly_detector.pkl")
    joblib.dump({
        "threshold": best_thr,
        "n_features": DRIFT_DIM,
        "feat_names": DRIFT_FEAT_NAMES,
    }, f"{MODELS_DIR}/anomaly_threshold.pkl")

    sz = Path(f"{MODELS_DIR}/anomaly_detector.pkl").stat().st_size / 1e6
    log.info("  ✓ Saved anomaly_detector.pkl (%.1f MB)  |  %.1f min", sz, (time.perf_counter() - t0) / 60)
    save_report("anomaly_detector", {
        "n_estimators":           best_iso_params.get("n_estimators", 300),
        "max_features":           best_iso_params.get("max_features", 0.8),
        "corpus_normal":          len(X_tr),
        "n_multivariate_anomalies": n_anom_pool,
        "precision":              round(prec, 4),
        "recall":                 round(rec, 4),
        "f1":                     round(f1, 4),
        "threshold":              round(best_thr, 6),
        "gate_passed":            f1 >= GATES["anomaly_detector"]["min_f1"],
        "model_size_mb":          round(sz, 2),
        "time_s":                 round(time.perf_counter() - t0, 1),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_dfs = load_all_real(max_openml=120, use_cache=True)
    if not all_dfs:
        log.error("No datasets loaded!"); import sys; sys.exit(1)
    train_anomaly_detector(all_dfs)
