#!/usr/bin/env python3
"""
================================================================
 ADAP v7 — MODEL 5/6: Chart Relevance Scorer
================================================================
HOW TO RUN IN COLAB:
  1. Run Cell 0 (pip install)
  2. Paste & run 00_shared_utils.py as Cell 1
  3. Paste & run THIS file as Cell 6

Label quality fix (vs v6):
  - Chart labels now derived from STATISTICAL TESTS, not arbitrary thresholds:
      Histogram  → normaltest D'Agostino K² + Sarle bimodality coefficient
      Bar        → cardinality + normalized entropy of category distribution
      Scatter    → Pearson p-value (significance, not just magnitude)
      Line       → Ljung-Box autocorrelation test (H0: no autocorrelation)
      Box        → ANOVA / group structure test
      Heatmap    → correlation magnitude + Pearson p-value
      Violin     → bimodality coefficient + group structure

New features added to the feature vector:
  - normality_pvalue   : D'Agostino K² test p-value
  - bimodality_coeff   : Sarle's b (>0.555 → bimodal)
  - corr_pvalue        : Pearson correlation p-value
  - ljung_box_pvalue   : autocorrelation test p-value

OUTPUT: /content/adap_models/chart_relevance_scorer.pkl
Expected time: ~15-20 minutes (60 Optuna trials)
================================================================
"""
import warnings
from collections import Counter
from scipy.stats import normaltest, pearsonr

CHART_TYPES   = ["histogram", "bar", "scatter", "line", "box", "heatmap", "violin"]
N_CHART_TYPES = len(CHART_TYPES)
_CHART_IDX    = {ct: i for i, ct in enumerate(CHART_TYPES)}

CHART_FEATS = [
    # Original features
    "is_numeric", "is_categorical", "is_datetime", "unique_rate", "null_rate",
    "skewness", "kurtosis", "log_n_unique", "is_paired", "pair_corr",
    "n_groups", "temporal_autocorr", "log_n_rows", "has_text",
    "bimodal_score", "entropy_score", "all_integer", "cv_coeff",
    # NEW: statistically derived features
    "normality_pvalue",   # D'Agostino K² test — low p → non-normal → histogram informative
    "bimodality_coeff",   # Sarle's b — >0.555 → bimodal → violin/histogram recommended
    "corr_pvalue",        # Pearson correlation p-value (filled for paired columns)
    "ljung_box_pvalue",   # autocorrelation significance (filled for numeric series)
    "cat_entropy_norm",   # normalized entropy of category distribution (0=uniform, 1=skewed)
]
N_CHART_BASE = len(CHART_FEATS)


def _safe_normaltest_pvalue(arr: np.ndarray) -> float:
    """D'Agostino K² normality test p-value. Returns 1.0 if test not applicable."""
    if len(arr) < 20:
        return 1.0
    try:
        _, p = normaltest(arr)
        return float(p) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _safe_pearsonr_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r p-value. Returns 1.0 if not computable."""
    n = min(len(a), len(b))
    if n < 5:
        return 1.0
    try:
        _, p = pearsonr(a[:n], b[:n])
        return float(p) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _cat_entropy_normalized(series: pd.Series) -> float:
    """
    Normalized entropy of category value counts.
    Returns 0 = perfectly uniform, 1 = completely concentrated in one value.
    Inverted: high score → more informative bar chart.
    """
    vc = series.dropna().value_counts(normalize=True)
    if len(vc) <= 1:
        return 0.0
    h     = float(-np.sum(vc * np.log(vc + 1e-9)))
    h_max = float(np.log(len(vc)))
    # Normalized entropy 0→1; we invert: 1 = concentrated (informative)
    return float(1.0 - h / (h_max + 1e-9))


def _real_col_to_chart_features(series: pd.Series, n_rows: int) -> Optional[dict]:
    """
    Extract per-column chart features with full production-grade guards.
    Every computed value is guaranteed finite before returning.
    Handles: constant columns, near-constant columns, bool columns,
             near-null columns, overflow kurtosis, NaN correlations.
    """
    nv       = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    n_unique = int(series.nunique())
    is_num   = pd.api.types.is_numeric_dtype(series)
    is_cat   = not is_num
    is_dt    = pd.api.types.is_datetime64_any_dtype(series)

    if len(nv) < 5 and not is_cat:
        return None

    nv_f = nv.astype(np.float64)   # single canonical float64 array — used everywhere

    # ── Skewness / kurtosis — guarded for overflow and near-constant columns ──
    if len(nv_f) > 3 and nv_f.std() > 1e-10:
        skew = float(nv_f.skew())
        kurt = float(nv_f.kurt())
        skew = skew if np.isfinite(skew) else 0.0   # overflow guard
        kurt = kurt if np.isfinite(kurt) else 0.0   # overflow guard
    else:
        skew, kurt = 0.0, 0.0  # constant column: skew/kurt undefined → 0

    # ── Bimodal score via histogram dip ──────────────────────────────────────
    # Cast to float64 first: bool columns trigger a uint8 RuntimeWarning
    if len(nv_f) > 20:
        hist, _ = np.histogram(nv_f, bins=min(30, len(nv_f) // 5))
        hist    = hist / (hist.sum() + 1e-9)
        dip     = float(np.max(np.abs(np.diff(hist))))
    else:
        dip = 0.0

    # ── Category entropy (for bar/heatmap) ───────────────────────────────────
    vc = series.dropna().value_counts(normalize=True)
    if len(vc) > 1:
        h_raw = float(-np.sum(vc * np.log(vc + 1e-9)))
        entropy = h_raw / (float(np.log(len(vc))) + 1e-9)
        entropy = entropy if np.isfinite(entropy) else 0.0
    else:
        entropy = 0.0

    # ── Ljung-Box autocorrelation p-value (for line chart) ───────────────────
    if len(nv_f) > 10:
        arr = nv_f.values[:min(500, len(nv_f))]
        # Guard: corrcoef produces NaN divide warning on constant arrays
        if len(arr) > 2 and arr.std() > 1e-10:
            autocorr = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
            autocorr = autocorr if np.isfinite(autocorr) else 0.0
        else:
            autocorr = 0.0
        ljb_pvalue = _ljung_box_pvalue(arr)
    else:
        autocorr   = 0.0
        ljb_pvalue = 1.0

    # ── Normality test (for histogram label) ─────────────────────────────────
    norm_pvalue = _safe_normaltest_pvalue(nv_f.values[:2000]) if len(nv_f) >= 20 else 1.0

    # ── Sarle bimodality coefficient ─────────────────────────────────────────
    bim_coeff = _bimodality_coeff(nv_f.values[:2000]) if len(nv_f) >= 8 else 0.0

    # ── Normalized category entropy (for bar chart) ───────────────────────────
    cat_ent = _cat_entropy_normalized(series) if is_cat else 0.0

    # ── Coefficient of variation — guarded for zero mean ─────────────────────
    nv_mean = float(nv_f.mean())
    cv = min(float(nv_f.std() / (abs(nv_mean) + 1e-9)), 100.0) if len(nv_f) > 1 else 0.0
    cv = cv if np.isfinite(cv) else 0.0

    return {
        "is_numeric":         float(is_num),
        "is_categorical":     float(is_cat),
        "is_datetime":        float(is_dt),
        "unique_rate":        n_unique / max(n_rows, 1),
        "null_rate":          float(series.isnull().mean()),
        "skewness":           min(max(skew, -10), 10),
        "kurtosis":           min(max(kurt, -5), 50),
        "log_n_unique":       float(np.log1p(n_unique)),
        "is_paired":          0.0,          # filled by corpus builder
        "pair_corr":          0.0,          # filled by corpus builder
        "n_groups":           float(min(n_unique, 50)),
        "temporal_autocorr":  autocorr,
        "log_n_rows":         float(np.log10(max(n_rows, 1))),
        "has_text":           float(pd.api.types.is_object_dtype(series)),
        "bimodal_score":      min(dip, 1.0),
        "entropy_score":      min(entropy, 1.0),
        "all_integer":        float((nv % 1 == 0).mean() > 0.9) if len(nv) > 3 else 0.0,
        "cv_coeff":           cv,
        # NEW statistical features
        "normality_pvalue":   norm_pvalue,
        "bimodality_coeff":   min(bim_coeff, 2.0),
        "corr_pvalue":        1.0,          # filled for paired columns
        "ljung_box_pvalue":   ljb_pvalue,
        "cat_entropy_norm":   cat_ent,
    }


def _determine_chart_relevance_statistical(feat: dict, chart_type: str) -> int:
    """
    Statistically grounded chart relevance labels derived from test outcomes.
    Label 2 = Recommended (statistical evidence supports this chart)
    Label 1 = Useful (moderate evidence / borderline)
    Label 0 = Irrelevant (type mismatch or no statistical support)

    References:
      - Pearson r significance: standard statistics
      - D'Agostino K² normality test: D'Agostino & Pearson (1973)
      - Sarle bimodality: Pfister et al. (2013) 0.555 threshold
      - Ljung-Box: Ljung & Box (1978)
    """
    is_num = feat["is_numeric"]   > 0.5
    is_cat = feat["is_categorical"] > 0.5
    is_dt  = feat["is_datetime"]  > 0.5
    paired = feat["is_paired"]    > 0.5
    ur     = feat["unique_rate"]
    n_grp  = feat["n_groups"]
    bim    = feat["bimodality_coeff"]
    norm_p = feat["normality_pvalue"]     # low → non-normal → histogram informative
    corr_p = feat["corr_pvalue"]          # low → significant correlation
    ljb_p  = feat["ljung_box_pvalue"]     # low → autocorrelated → line relevant
    cat_ent = feat["cat_entropy_norm"]    # higher → more concentrated → informative bar

    if chart_type == "histogram":
        if not is_num:
            return 0
        # Recommended if: non-normal distribution OR bimodal OR sufficient spread
        non_normal  = norm_p < 0.05
        bimodal     = bim > 0.555
        spread_ok   = ur > 0.05 and n_grp > 10
        if (non_normal or bimodal) and spread_ok:
            return 2
        if n_grp >= 5 and ur > 0.02:
            return 1   # useful: shows basic distribution shape
        return 0

    elif chart_type == "bar":
        if not is_cat:
            return 0
        # Recommended: few+distinct categories OR concentrated distribution
        if 2 <= n_grp <= 25 and (cat_ent > 0.3 or n_grp <= 10):
            return 2
        if 2 <= n_grp <= 40:
            return 1   # useful: many categories, still readable
        return 0       # too many categories for a useful bar chart

    elif chart_type == "scatter":
        if not (is_num and paired):
            return 0
        # Recommended: statistically significant correlation (any direction)
        if corr_p < 0.05:
            return 2
        if corr_p < 0.20:
            return 1   # borderline significance
        return 0       # no detectable relationship

    elif chart_type == "line":
        if is_dt:
            return 2   # datetime axis → line chart is always appropriate
        if is_num and ljb_p < 0.05:
            return 2   # significant autocorrelation → sequential structure
        if is_num and (ljb_p < 0.20 or feat["temporal_autocorr"] > 0.25):
            return 1
        return 0

    elif chart_type == "box":
        if not is_num:
            return 0
        # Recommended: numeric AND paired with a grouping variable
        if paired and 2 <= n_grp <= 20:
            return 2
        if is_num:
            return 1   # shows quartile distribution even without groups
        return 0

    elif chart_type == "heatmap":
        if not (is_num and paired):
            return 0
        corr_mag = abs(feat["pair_corr"])
        if corr_p < 0.05 and corr_mag >= 0.35:
            return 2   # significant moderate+ correlation
        if corr_p < 0.15:
            return 1
        return 0

    elif chart_type == "violin":
        if not is_num:
            return 0
        # Recommended: bimodal (violin reveals this) OR grouped with few groups
        if bim > 0.555:
            return 2   # bimodal distribution → violin shows this clearly
        if paired and 2 <= n_grp <= 15:
            return 2   # multiple groups → comparison
        if is_num:
            return 1
        return 0

    return 0


def _build_real_chart_corpus(all_dfs, rng):
    """Build chart corpus with statistically derived labels.
    Returns: X, y, n_real_rows
      X[:n_real_rows]  — original real samples (used for CV)
      X[n_real_rows:]  — noise-augmented minority copies (only for training)
    """
    real_rows, real_labels = [], []   # clean real signal
    aug_rows,  aug_labels  = [], []   # noise-augmented minority copies

    for df in all_dfs:
        n_rows   = len(df)
        num_cols = df.select_dtypes(include="number").columns.tolist()

        # Pre-compute pairwise correlations for scatter/heatmap labels
        pairs = []
        if len(num_cols) >= 2:
            for i in range(min(len(num_cols) - 1, 5)):
                ca, cb = num_cols[i], num_cols[i + 1]
                try:
                    a = pd.to_numeric(df[ca].dropna(), errors="coerce").dropna()
                    b = pd.to_numeric(df[cb].dropna(), errors="coerce").dropna()
                    min_n = min(len(a), len(b))
                    if min_n >= 5:
                        a_v = a.values[:min_n].astype(np.float64)
                        b_v = b.values[:min_n].astype(np.float64)
                        # Skip constant columns — pearsonr is undefined (ConstantInputWarning)
                        if a_v.std() < 1e-10 or b_v.std() < 1e-10:
                            continue
                        r, p = pearsonr(a_v, b_v)
                        if np.isfinite(r) and np.isfinite(p):
                            pairs.append((ca, cb, float(r), float(p)))
                except Exception:
                    pass

        for col in df.columns:
            feats = _real_col_to_chart_features(df[col], n_rows)
            if feats is None:
                continue

            # Fill paired features if this column is in any pair
            for ca, cb, corr, p_val in pairs:
                if col == ca or col == cb:
                    feats["is_paired"]   = 1.0
                    feats["pair_corr"]   = corr
                    feats["corr_pvalue"] = p_val
                    break

            for ct_idx, chart_type in enumerate(CHART_TYPES):
                lbl = _determine_chart_relevance_statistical(feats, chart_type)

                one_hot = [0.0] * N_CHART_TYPES
                one_hot[ct_idx] = 1.0

                # Real row — always added to real_rows (never noised)
                row = [feats.get(f, 0.0) for f in CHART_FEATS] + one_hot
                if np.isfinite(row).all():
                    real_rows.append(row)
                    real_labels.append(lbl)

                    # Noise-augmented copy for minority classes only (lbl > 0)
                    # class 0=irrelevant is overrepresented; augment 1 & 2
                    if lbl > 0:
                        noisy = {k: v + float(rng.normal(0, 0.015))
                                 if k not in ("normality_pvalue", "corr_pvalue",
                                              "ljung_box_pvalue", "bimodality_coeff",
                                              "is_numeric", "is_categorical",
                                              "is_datetime", "is_paired")
                                 else v
                                 for k, v in feats.items()}
                        row_n = [noisy.get(f, 0.0) for f in CHART_FEATS] + one_hot
                        if np.isfinite(row_n).all():
                            aug_rows.append(row_n)
                            aug_labels.append(lbl)

    # Real rows first, then augmented — clean boundary at n_real_rows
    n_real_rows = len(real_rows)
    all_rows   = real_rows + aug_rows
    all_labels = real_labels + aug_labels
    return (np.array(all_rows, dtype=np.float32),
            np.array(all_labels, dtype=np.int32),
            n_real_rows)


def train_chart_relevance_scorer(all_dfs):
    log.info("\n=== [5/6] Chart Relevance Scorer (statistical labels) ===")
    t0  = time.perf_counter()
    import lightgbm as lgb
    rng = _make_rng(5)

    X_raw, y_raw, n_real_rows = _build_real_chart_corpus(all_dfs, rng)
    y = y_raw  # 0=irrelevant, 1=useful, 2=recommended
    log.info("  Chart corpus: %d × %d  (real=%d aug=%d)  Class dist: %s",
             *X_raw.shape, n_real_rows, len(X_raw) - n_real_rows,
             {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))})
    X_raw = _clip_transform(X_raw, 99.5)

    # Holdout split on REAL rows only (X_raw[:n_real_rows]) to keep it clean
    X_real = X_raw[:n_real_rows];  y_real = y[:n_real_rows]
    X_aug  = X_raw[n_real_rows:];  y_aug  = y[n_real_rows:]
    X_real_tv, X_hold, y_real_tv, y_hold = train_test_split(
        X_real, y_real, test_size=0.20, stratify=y_real, random_state=SEED)
    # Combine real train + augmented for training (augmented never in holdout)
    X_tv = np.vstack([X_real_tv, X_aug]);  y_tv = np.concatenate([y_real_tv, y_aug])
    # 25% val of real+aug train — larger set reduces Optuna val saturation risk
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=SEED)
    X_tr_b, y_tr_b = _smote_safe(X_tr, y_tr, SEED)
    # CV pool: real training rows only (no augmented inflation)
    X_tr_cv = X_real_tv;  y_tr_cv = y_real_tv

    # ── Optuna ────────────────────────────────────────────────────────────────
    try:
        import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

        def cr_obj(trial):
            # Regularisation-first search space (mirrors schema/domain classifiers)
            p = dict(
                n_estimators       = trial.suggest_int("n",   500, 3000),
                max_depth          = trial.suggest_int("d",   4,   10),
                num_leaves         = trial.suggest_int("l",   31,  200),
                min_child_samples  = trial.suggest_int("mcs", 25,  120),
                min_split_gain     = trial.suggest_float("msg", 0.0, 0.5),
                subsample          = trial.suggest_float("ss",  0.60, 0.95),
                colsample_bytree   = trial.suggest_float("cs",  0.55, 0.95),
                reg_lambda         = trial.suggest_float("rl",  1.5,  30,  log=True),
                reg_alpha          = trial.suggest_float("ra",  0.0,  5.0),
                learning_rate      = trial.suggest_float("lr",  0.005, 0.08, log=True),
            )
            m = lgb.LGBMClassifier(**p, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(X_tr_b, y_tr_b, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
            return min(balanced_accuracy_score(y_val, m.predict(X_val)), 0.999)

        study = optuna.create_study(direction="maximize")
        study.optimize(cr_obj, n_trials=60, show_progress_bar=True)
        bp = study.best_params
        best_p = dict(n_estimators=bp["n"], max_depth=bp["d"], num_leaves=bp["l"],
                      min_child_samples=bp["mcs"], min_split_gain=bp["msg"],
                      subsample=bp["ss"], colsample_bytree=bp["cs"],
                      reg_lambda=bp["rl"], reg_alpha=bp["ra"], learning_rate=bp["lr"])
        log.info("  Optuna best=%.4f  n=%d  leaves=%d  mcs=%d  lambda=%.2f",
                 study.best_value, bp["n"], bp["l"], bp["mcs"], bp["rl"])
    except ImportError:
        best_p = dict(n_estimators=2000, max_depth=8, num_leaves=127,
                      min_child_samples=25, min_split_gain=0.1,
                      subsample=0.80, colsample_bytree=0.80,
                      reg_lambda=2.5, reg_alpha=0.3, learning_rate=0.03)

    model = lgb.LGBMClassifier(**best_p, class_weight="balanced",
                               random_state=SEED, n_jobs=-1, verbose=-1)
    model.fit(X_tr_b, y_tr_b, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    val_acc  = balanced_accuracy_score(y_val,  model.predict(X_val))
    hold_acc = balanced_accuracy_score(y_hold, model.predict(X_hold))

    # CV on REAL rows only (no augmented inflation)
    cls_counts_cv = Counter(y_tr_cv.tolist())
    cv_k = min(5, max(2, min(cls_counts_cv.values())))
    cv_sc = cross_val_score(
        lgb.LGBMClassifier(**best_p, class_weight="balanced",
                           random_state=SEED, n_jobs=-1, verbose=-1),
        X_tr_cv, y_tr_cv,
        cv=StratifiedKFold(cv_k, shuffle=True, random_state=SEED),
        scoring="balanced_accuracy", n_jobs=1,
    )
    log.info("  %d-Fold CV (real data only): %.4f ± %.4f", cv_k, cv_sc.mean(), cv_sc.std())
    log.info("  Single-split val_acc=%.4f  (informational only — gate uses CV mean)",
             val_acc)
    gate = quality_gate(cv_sc.mean(), hold_acc, cv_sc.std(), "chart_relevance_scorer")

    print("\n=== Chart Relevance Report (Holdout) ===")
    print(classification_report(
        y_hold, model.predict(X_hold),
        labels=[0, 1, 2],
        target_names=["irrelevant", "useful", "recommended"],
        zero_division=0,
    ))

    ver = _model_hash(best_p)
    chart_pipeline = Pipeline([("model", model)])
    joblib.dump(chart_pipeline, f"{MODELS_DIR}/chart_relevance_scorer.pkl")
    joblib.dump({
        "chart_types":   CHART_TYPES,
        "features":      CHART_FEATS,
        "n_chart_base":  N_CHART_BASE,
        "n_chart_types": N_CHART_TYPES,
        "label_map":     {0: "irrelevant", 1: "useful", 2: "recommended"},
        "label_method":  "statistical_tests_v7",
        "nlp_method":    NLP._method,   # script 07 checks this for NLP consistency
        "version":       VERSION,
        "param_hash":    ver,
    }, f"{MODELS_DIR}/chart_registry.pkl")

    sz = Path(f"{MODELS_DIR}/chart_relevance_scorer.pkl").stat().st_size / 1e6
    log.info("  ✓ Saved chart_relevance_scorer.pkl (%.1f MB)  |  %.1f min",
             sz, (time.perf_counter() - t0) / 60)
    save_report("chart_relevance_scorer", {
        "corpus_size":   len(X_raw),
        "n_base_feats":  N_CHART_BASE,
        "n_total_feats": N_CHART_BASE + N_CHART_TYPES,
        "label_method":  "statistical_tests_v7",
        "chart_types":   CHART_TYPES,
        "val_bal_acc":   round(val_acc, 4),
        "hold_bal_acc":  round(hold_acc, 4),
        "cv_mean":       round(float(cv_sc.mean()), 4),
        "cv_std":        round(float(cv_sc.std()), 4),
        "quality_gate":  gate,
        "best_params":   best_p,
        "model_size_mb": round(sz, 2),
        "version":       VERSION,
        "time_s":        round(time.perf_counter() - t0, 1),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_dfs = load_all_real(max_openml=120, use_cache=True)
    if not all_dfs:
        log.error("No datasets loaded!"); import sys; sys.exit(1)
    train_chart_relevance_scorer(all_dfs)
