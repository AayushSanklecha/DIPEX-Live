# =========================================================================
# DIPEX — Schema Classifier Patch: Exotic Semantic Types
# =========================================================================
# INSTRUCTIONS: Run this as Cell 5b in Colab, AFTER Cell 5 & BEFORE Cell 7.
#
# This patch EXTENDS the already-trained schema classifier with NEW
# exotic semantic types that real-world enterprise data can contain:
#
#   swift_code       — BIC/SWIFT bank codes (e.g. "HDFCINBB", "BOFAUS3N")
#   iban             — International Bank Account Numbers
#   ssn              — US Social Security Numbers ("123-45-6789")
#   pan_number       — Indian PAN ("ABCDE1234F")
#   passport         — Passport numbers (alphanumeric, 8-9 chars)
#   vin              — Vehicle Identification Numbers (17-char)
#   mac_address      — Network MAC addresses
#   credit_card      — Masked/full credit card numbers
#   ticker_symbol    — Stock tickers (e.g. "GOOG", "MSFT", "RELIANCE.NS")
#   hash_value       — MD5/SHA256 hashes
#
# Strategy: IncrementalPatch — generates synthetic samples for ONLY the
# new labels, extracts the same 25-feature vector, and trains new trees
# via RandomForestClassifier warm_start, then re-serialises the combined
# model artifact.
# =========================================================================

import os, warnings, logging
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
log = logging.getLogger("dipex_schema_patch")
logging.basicConfig(level=logging.INFO, format="%(message)s")

MODELS_DIR = "/content/dipex_models"
RNG = np.random.default_rng(2025)

# ── New exotic label definitions ─────────────────────────────────────────
EXOTIC_LABELS = [
    "swift_code", "iban", "ssn", "pan_number",
    "passport", "vin", "mac_address", "credit_card",
    "ticker_symbol", "hash_value"
]

def _make_exotic_series(label: str, n: int) -> pd.Series:
    null_p = RNG.uniform(0.0, 0.30)

    def _null(s: pd.Series) -> pd.Series:
        s = s.copy()
        if null_p > 0:
            idx = RNG.choice(len(s), max(1, int(len(s) * null_p)), replace=False)
            s.iloc[idx] = np.nan
        return s

    if label == "swift_code":
        # BIC format: 4 alpha (bank) + 2 alpha (country) + 2 alpha/num (location) + optional 3
        banks = ["HDFC", "BOFA", "CITI", "HSBC", "DEUT", "BNPA", "ICIC", "AXIS", "WELL"]
        countries = ["IN", "US", "GB", "DE", "FR", "SG", "AE", "AU", "JP"]
        locs = ["BB", "3N", "LN", "FF", "PP", "HH", "MM", "TT"]
        arr = [f"{RNG.choice(banks)}{RNG.choice(countries)}{RNG.choice(locs)}" for _ in range(n)]
        # Add 3-char branch code to some (full 11-char)
        arr = [a + str(RNG.choice(["XXX", "001", "002", ""])) if RNG.random() > 0.5 else a for a in arr]
        return _null(pd.Series(arr))

    elif label == "iban":
        # Simplified IBAN: 2-char country + 2 check digits + 10-28 alphanumeric
        countries = ["GB", "DE", "FR", "IN", "AE", "US", "NL", "IT", "ES"]
        arr = [f"{RNG.choice(countries)}{RNG.integers(10,99)}" +
               "".join([str(RNG.integers(0, 10)) for _ in range(RNG.integers(10, 20))])
               for _ in range(n)]
        return _null(pd.Series(arr))

    elif label == "ssn":
        # Format: XXX-XX-XXXX
        arr = [f"{RNG.integers(100,999)}-{RNG.integers(10,99)}-{RNG.integers(1000,9999)}"
               for _ in range(n)]
        return _null(pd.Series(arr))

    elif label == "pan_number":
        # Indian PAN: 5 alpha + 4 digits + 1 alpha (e.g. ABCDE1234F)
        alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        arr = [
            "".join(RNG.choice(list(alpha), 5).tolist()) +
            "".join([str(RNG.integers(0, 10)) for _ in range(4)]) +
            str(RNG.choice(list(alpha)))
            for _ in range(n)
        ]
        return _null(pd.Series(arr))

    elif label == "passport":
        # Typically 1-2 letters + 7 numbers (varies by country)
        alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        arr = [
            "".join(RNG.choice(list(alpha), RNG.integers(1, 3)).tolist()) +
            "".join([str(RNG.integers(0, 10)) for _ in range(7)])
            for _ in range(n)
        ]
        return _null(pd.Series(arr))

    elif label == "vin":
        # VIN: exactly 17 alphanumeric chars (no I, O, Q)
        vin_chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
        arr = ["".join(RNG.choice(list(vin_chars), 17).tolist()) for _ in range(n)]
        return _null(pd.Series(arr))

    elif label == "mac_address":
        # Format: XX:XX:XX:XX:XX:XX or XX-XX-XX-XX-XX-XX
        sep = RNG.choice([":", "-"])
        arr = [sep.join([f"{RNG.integers(0, 256):02X}" for _ in range(6)]) for _ in range(n)]
        return _null(pd.Series(arr))

    elif label == "credit_card":
        # Masked: XXXX-XXXX-XXXX-1234  or full 16-digit number
        choice = RNG.integers(0, 3)
        if choice == 0:
            # Full number
            arr = [str(RNG.integers(4000000000000000, 4999999999999999)) for _ in range(n)]
        elif choice == 1:
            # Masked
            arr = [f"****-****-****-{RNG.integers(1000,9999)}" for _ in range(n)]
        else:
            # Grouped
            arr = [f"{RNG.integers(4000,4999)}-{RNG.integers(1000,9999)}-{RNG.integers(1000,9999)}-{RNG.integers(1000,9999)}"
                   for _ in range(n)]
        return _null(pd.Series(arr))

    elif label == "ticker_symbol":
        # Mix of US tickers (1-5 alpha) and Indian tickers (NAME.NS / NAME.BO)
        us = ["AAPL", "GOOG", "MSFT", "TSLA", "NVDA", "AMZN", "META", "BRK", "JPM", "V"]
        in_tickers = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "WIPRO.BO", "IRCTC.NS"]
        pool = us + in_tickers
        arr = RNG.choice(pool, n).tolist()
        return _null(pd.Series(arr))

    elif label == "hash_value":
        # MD5 (32 hex) or SHA256 (64 hex)
        import hashlib, uuid
        choice = RNG.integers(0, 2)
        if choice == 0:
            arr = [hashlib.md5(str(RNG.integers(0, 999999)).encode()).hexdigest() for _ in range(n)]
        else:
            arr = [hashlib.sha256(str(RNG.integers(0, 999999)).encode()).hexdigest() for _ in range(n)]
        return _null(pd.Series(arr))

    return pd.Series(RNG.normal(0, 1, n))


# ── Feature extraction (same 25-feature vector as main script) ───────────
def robust_extract_features(s: pd.Series) -> list:
    n_total = max(len(s), 1)
    s_clean = s.dropna()
    is_num = pd.api.types.is_numeric_dtype(s)
    is_str = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)

    num_vals = pd.to_numeric(s_clean, errors='coerce').dropna() if not is_num else s_clean.copy()
    str_vals = s_clean.astype(str) if is_str else pd.Series([], dtype=str)
    num_vals = num_vals[~np.isinf(num_vals)]

    try: all_int = float((num_vals == num_vals.apply(int)).all()) if len(num_vals)>0 else 0.0
    except: all_int = 0.0

    return [
        float(s.isnull().mean()),
        float(s_clean.nunique() / n_total),
        float(is_num), float(is_str),
        float(pd.api.types.is_datetime64_any_dtype(s)),
        float(num_vals.mean() if len(num_vals) else 0.0),
        float(num_vals.std()  if len(num_vals) else 0.0),
        float(num_vals.min()  if len(num_vals) else 0.0),
        float(num_vals.max()  if len(num_vals) else 0.0),
        float(num_vals.skew() if len(num_vals)>3 else 0.0),
        all_int,
        float(num_vals.max() < 200  if len(num_vals) else 0.0),
        float(num_vals.max() <= 1.0 if len(num_vals) else 0.0),
        float((num_vals >= 0).all() if len(num_vals) else 0.0),
        float(s_clean.nunique()),
        float(str_vals.str.contains(r"@.*\.", na=False).mean() if len(str_vals) else 0),
        float(str_vals.str.contains(r"^\+?\d[\d\s\-()]{7,}$", na=False).mean() if len(str_vals) else 0),
        float(str_vals.str.len().mean() if len(str_vals) else 0),
        float(s_clean.nunique()/n_total > 0.9),
        float(s_clean.nunique()/n_total < 0.05),
        float(str_vals.str.contains(r"https?://|www\.", na=False).mean() if len(str_vals) else 0),
        float(str_vals.str.match(r"^(\d{1,3}\.){3}\d{1,3}$", na=False).mean() if len(str_vals) else 0),
        float(((num_vals >= -180) & (num_vals <= 180)).all() if len(num_vals) else 0),
        float((num_vals % 1 != 0).mean() > 0.8 if len(num_vals) else 0),
        float(str_vals.str.match(r"^[A-Z]{3}$", na=False).mean() > 0.7 if len(str_vals) else 0),
    ]


# ── Generate exotic training data ─────────────────────────────────────────
def generate_exotic_data(n_per_class=500):
    X_list, y_list = [], []
    log.info("\n[PATCH] Generating exotic type samples...")
    for lbl in EXOTIC_LABELS:
        for _ in range(n_per_class):
            try:
                s = _make_exotic_series(lbl, int(RNG.integers(50, 400)))
                X_list.append(robust_extract_features(s))
                y_list.append(lbl)
            except: pass
        log.info(f"  [+] {lbl:<20}: {n_per_class} samples")
    return (
        np.nan_to_num(np.array(X_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
        np.array(y_list),
    )


# ── Patch: Extend LabelEncoder + retrain combined forest ─────────────────
def patch_schema_classifier():
    log.info("\n" + "="*70)
    log.info(" PATCHING SCHEMA CLASSIFIER WITH 10 EXOTIC TYPES")
    log.info("="*70)

    # 1. Load existing artifacts
    clf_old = joblib.load(os.path.join(MODELS_DIR, "schema_classifier.pkl"))
    le_old  = joblib.load(os.path.join(MODELS_DIR, "schema_label_encoder.pkl"))

    log.info(f"  Loaded existing model with {len(le_old.classes_)} classes.")

    # 2. Generate exotic samples
    X_new, y_new_str = generate_exotic_data(n_per_class=500)

    # 3. Rebuild the LabelEncoder to include BOTH old + new labels
    all_classes = list(le_old.classes_) + [l for l in EXOTIC_LABELS if l not in le_old.classes_]
    le_new = LabelEncoder()
    le_new.fit(all_classes)

    log.info(f"\n  LabelEncoder extended: {len(le_old.classes_)} → {len(le_new.classes_)} classes")

    # 4. Re-encode existing predictions space using a compatibility bridge
    #    We can't retrain from scratch (no original data), so we build a NEW
    #    forest trained ONLY on the patch data, then combine both into a
    #    VotingClassifier-like single refit model.
    #
    #    BEST STRATEGY: Train a new small RF on exotic data ONLY, then combine
    #    into a meta-forest using the combined LabelEncoder output space.
    #
    #    NOTE: sklearn RF does not support partial_fit. So we train a patch-only
    #    RF and route predictions through a smart combiner at inference time.

    y_new_enc = le_new.transform(y_new_str)

    clf_patch = RandomForestClassifier(
        n_estimators=200, max_depth=14, min_samples_leaf=4,
        class_weight="balanced", n_jobs=-1, random_state=42
    )
    clf_patch.fit(X_new, y_new_enc)
    patch_acc = clf_patch.score(X_new, y_new_enc)
    log.info(f"\n  Patch RF Training Accuracy (own space): {patch_acc:.3f}")

    # 5. Wrap both models into a single smart combiner and re-serialize
    #    The combiner checks if prediction confidence < threshold → falls back to patch RF
    class CombinedSchemaClassifier:
        """
        Smart two-stage schema classifier:
          Stage 1: Original model (21 known types) — high confidence when score > threshold
          Stage 2: Exotic patch model (10 exotic types) — handles what Stage 1 is uncertain about
        """
        def __init__(self, clf_base, clf_patch, le_base, le_new, exotic_labels, threshold=0.55):
            self.clf_base      = clf_base
            self.clf_patch     = clf_patch
            self.le_base       = le_base
            self.le_new        = le_new
            self.exotic_labels = exotic_labels
            self.threshold     = threshold

        def predict(self, X):
            # Get base probabilities
            try:
                proba_base = self.clf_base.predict_proba(X)
                max_conf   = proba_base.max(axis=1)
                base_preds = self.le_base.inverse_transform(self.clf_base.predict(X))
            except Exception:
                base_preds = np.array(["unknown"] * len(X))
                max_conf   = np.zeros(len(X))

            # For low-confidence rows, try the patch model
            results = base_preds.copy().astype(object)
            low_conf_mask = max_conf < self.threshold

            if low_conf_mask.any():
                X_low = X[low_conf_mask]
                try:
                    patch_enc  = self.clf_patch.predict(X_low)
                    patch_preds = self.le_new.inverse_transform(patch_enc)
                    results[low_conf_mask] = patch_preds
                except Exception:
                    pass  # Keep base prediction on failure

            return results

        def predict_proba_best(self, X):
            """Returns confidence score for the chosen prediction."""
            try:
                proba_base = self.clf_base.predict_proba(X)
                return proba_base.max(axis=1)
            except:
                return np.zeros(len(X))

    combined = CombinedSchemaClassifier(
        clf_base=clf_old, clf_patch=clf_patch,
        le_base=le_old, le_new=le_new,
        exotic_labels=EXOTIC_LABELS,
        threshold=0.55  # If base confidence < 55%, defer to patch model
    )

    # 6. Save the extended model and new LabelEncoder
    joblib.dump(combined, os.path.join(MODELS_DIR, "schema_classifier.pkl"))
    joblib.dump(le_new,   os.path.join(MODELS_DIR, "schema_label_encoder.pkl"))

    log.info("\n  ✓ Patch complete!")
    log.info(f"  ✓ schema_classifier.pkl now recognises {len(le_new.classes_)} semantic types:")
    for i, cls in enumerate(sorted(le_new.classes_)):
        marker = " (NEW)" if cls in EXOTIC_LABELS else ""
        log.info(f"      [{i:02d}]  {cls}{marker}")
    log.info("\n  Download the updated schema_classifier.pkl and schema_label_encoder.pkl")


# ── Run patch ─────────────────────────────────────────────────────────────
# Auto-bootstrap: if the base model doesn't exist yet (Cell 7 not run),
# call train_schema() from the Colab session so this cell is self-sufficient.
_schema_pkl = os.path.join(MODELS_DIR, "schema_classifier.pkl")

if not os.path.exists(_schema_pkl):
    log.info("[BOOTSTRAP] schema_classifier.pkl not found.")
    log.info("[BOOTSTRAP] Attempting to call train_schema() from current session...")
    try:
        train_schema()  # Defined in Cell 5 — must be in Colab session memory
        log.info("[BOOTSTRAP] train_schema() completed. Proceeding with patch.")
    except NameError:
        log.error("="*70)
        log.error("ERROR: train_schema() is not defined in this session.")
        log.error("You must run Cell 5 first to define and save the base model.")
        log.error("Run order: Cell 5 → Cell 5b → Cell 5c")
        log.error("="*70)
        raise RuntimeError("Run Cell 5 before Cell 5b. See instructions above.")

patch_schema_classifier()
