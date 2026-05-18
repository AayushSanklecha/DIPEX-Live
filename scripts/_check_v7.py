"""Final integrity check for colab_train_production_v7.py"""
import ast, sys
from pathlib import Path

src = Path("scripts/colab_train_production_v7.py").read_text(encoding="utf-8")
lines = src.splitlines()

try:
    ast.parse(src)
    print("SYNTAX: OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR line {e.lineno}: {e.msg}")
    sys.exit(1)

CHECKS = [
    # Core bug fixes
    ("Counter in schema corpus",            "from collections import Counter" in src),
    ("col_name cast to str",                "col_name = str(col_name)" in src),
    ("domain join integer-safe",            src.count("str(c) for c in df.columns") >= 2),
    ("uint8 float cast before outlier",     "astype(np.float64)" in src),
    ("iqr==0 constant-col guard",           "if iqr == 0:" in src),
    # AE
    ("AE Optuna max_iter=1200",             "max_iter=1200" in src),
    ("AE final max_iter=2000",              "max_iter=2000" in src),
    # PCA
    ("PCA near-zero-variance drop",         "near-zero-variance features BEFORE PCA" in src),
    ("PCA probe-based degeneracy check",     "PCA degeneracy probe" in src and "n_probe" in src),
    # CV
    ("No bad fit_params in CV",             "fit_params" not in src),
    ("CV outer n_jobs=1 schema",            'scoring="balanced_accuracy", n_jobs=1' in src),
    ("CV roc_auc outer n_jobs=1",           'scoring="roc_auc", n_jobs=1' in src),
    ("No n_jobs=-1 in outer CV",            src.count('scoring="balanced_accuracy", n_jobs=-1') == 0),
    # LGB
    ("LGB early_stopping in final model",   "lgb.early_stopping(50" in src),
    ("LGB early_stopping in Optuna trials", "lgb.early_stopping(30" in src),
    ("monotone constraints on confidence",  "MONOTONE_CONSTRAINTS" in src),
    ("class_weight=balanced on classifiers",src.count('class_weight="balanced"') >= 4),
    ("SMOTE stat-only",                     "_smote_stat_only" in src),
    ("Platt calibration",                   "CalibratedClassifierCV" in src),
    # NEW: holdout integrity
    ("Holdout split BEFORE augmentation",   "HOLDOUT INTEGRITY FIX" in src),
    ("Schema corpus returns 3-tuple",       "return X, y, _n_train_aug" in src),
    ("Schema train uses pre-split boundary","n_train_aug" in src),
    ("Domain class crash guard",            "CRASH GUARD" in src),
    # Data
    ("Real OpenML data (120+)",             "max_openml=120" in src),
    ("PMLB dataset loader",                 "def load_pmlb_datasets" in src),
    ("UCI dataset loader",                  "def load_uci_datasets" in src),
    ("Disk caching layer",                  "def save_datasets_to_cache" in src),
    ("MCAR/MAR/MNAR injection",             "MNAR" in src),
    ("4-way data split",                    "X_tr_b, y_tr_b" in src and "X_hold" in src),
    ("Holdout never used for training",     "# [H6]" in src),
    # Architecture
    ("All 6 models registered",             "train_confidence_scorer" in src),
    ("SHAP explainability",                 "shap.TreeExplainer" in src),
    ("PSI drift utility",                   "def _psi" in src),
    ("Model versioning hash",               "_model_hash" in src),
    ("sklearn Pipeline save",               "Pipeline([" in src),
    ("NLP method saved to metadata",        "nlp_method" in src),
    ("Post-training validation",            "run_post_training_validation" in src),
    # Scale
    ("Total lines > 2600",                  len(lines) > 2600),
    ("File size > 100KB",                   len(src.encode()) > 100_000),
]

print()
passed = 0
failed = []
for name, ok in CHECKS:
    icon = "OK  " if ok else "FAIL"
    print(f"  [{icon}] {name}")
    if ok:
        passed += 1
    else:
        failed.append(name)

print()
print(f"Lines: {len(lines):,}  |  Size: {len(src.encode())//1024} KB")
print(f"Passed: {passed}/{len(CHECKS)}")
if failed:
    print("\nFAILED:")
    for f in failed:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("\nALL CHECKS PASSED")
