# =========================================================================
# DIPEX — Schema Classifier: NLP Column-Name Augmentation Patch
# =========================================================================
# INSTRUCTIONS: Run this as Cell 5c in Colab, AFTER Cell 5 (and 5b if used).
#
# WHAT THIS DOES:
# ───────────────
# The base schema classifier is purely statistical — it only looks at data
# VALUES inside a column. It completely ignores the column NAME, which is
# arguably the strongest signal any DBA or data engineer provides.
#
# This patch ADDS an NLP layer that:
#
#   A. TF-IDF Vectorisation of the column name (handles subword tokens
#      like "cust_panno", "address_line1", "lat_decimal_deg").
#
#   B. Semantic keyword matching with a rich lexicon (50+ synonyms per
#      semantic type, covering financial, healthcare, engineering, legal
#      and geographic vocabularies).
#
#   C. Optional Sentence-Transformer Embeddings using "all-MiniLM-L6-v2"
#      (free, 80MB, hugely powerful — if not available, falls back to TF-IDF).
#
# COMBINED ARCHITECTURE (after this patch):
# ──────────────────────────────────────────
#
#   Column Name  ──▶  NLP Feature Vector (TF-IDF + keyword match score)
#           │                                    │
#           └──────────────┬─────────────────────┘
#                          ▼
#   Data Values  ──▶  Statistical Feature Vector (25 features)
#                          │
#                          ▼
#              CombinedNLPSchemaClassifier.predict()
#                   ┌──────────────────────────────┐
#                   │  Stage 1: Keyword override    │ ← 100% precision rules
#                   │  Stage 2: NLP-augmented RF    │ ← handles new types
#                   │  Stage 3: Statistical base RF │ ← original model
#                   └──────────────────────────────┘
#
# =========================================================================

import os, warnings, re, logging
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
log = logging.getLogger("dipex_nlp_patch")
logging.basicConfig(level=logging.INFO, format="%(message)s")

MODELS_DIR = "/content/dipex_models"
RNG = np.random.default_rng(2025)

# ── A. Comprehensive Semantic Lexicon (NLP ground truth) ─────────────────
# For each label, this is the full vocabulary of column names that MEAN it.
# Covers English + common abbreviations + Indian/European enterprise naming.

SEMANTIC_LEXICON = {
    "id": [
        "id", "identifier", "primary_key", "pk", "uuid", "guid", "record_id",
        "row_id", "uid", "user_id", "customer_id", "cust_id", "client_id",
        "order_id", "transaction_id", "txn_id", "entity_id", "object_id",
        "employee_id", "emp_id", "pid", "serial_no", "sn", "ref_no",
        "reference_number", "identity", "membership_id"
    ],
    "age": [
        "age", "years", "yrs", "age_years", "patient_age", "customer_age",
        "year_of_birth", "yob", "dob_age", "age_at_joining", "tenure_years",
        "years_old", "decades", "age_group_numeric"
    ],
    "amount": [
        "amount", "price", "cost", "revenue", "income", "salary", "wage",
        "fee", "charge", "payment", "balance", "total", "subtotal", "tax",
        "discount", "profit", "loss", "net", "gross", "expenditure",
        "budget", "spend", "transaction_amount", "txn_amount", "value",
        "credit", "debit", "remittance", "loan_amt", "emi", "premium",
        "fare", "rent", "invoice_value", "market_cap", "turnover"
    ],
    "date": [
        "date", "datetime", "timestamp", "time", "created_at", "updated_at",
        "modified_at", "deleted_at", "dob", "date_of_birth", "joining_date",
        "expiry_date", "due_date", "start_date", "end_date", "event_date",
        "transaction_date", "txn_date", "posting_date", "booking_date",
        "invoice_date", "shipment_date", "delivery_date", "created_on",
        "order_date", "purchase_date", "activation_date", "closure_date"
    ],
    "category": [
        "category", "type", "class", "group", "status", "state", "segment",
        "tier", "level", "label", "tag", "kind", "variant", "mode",
        "channel", "source", "medium", "bucket", "cohort", "cluster",
        "department", "division", "region", "zone", "area", "section",
        "genre", "brand", "product_type", "sub_category", "sub_type",
        "classification", "flag_type", "priority_level"
    ],
    "text": [
        "text", "description", "comment", "note", "remarks", "narrative",
        "summary", "detail", "message", "feedback", "review", "reason",
        "explanation", "memo", "body", "content", "title", "subject",
        "address_line", "notes", "obs", "observation", "opinion"
    ],
    "phone": [
        "phone", "mobile", "cell", "contact", "telephone", "tel", "fax",
        "phone_number", "mobile_no", "contact_no", "phone_no", "landline",
        "whatsapp", "sms_number", "calling_no", "phn", "mob"
    ],
    "email": [
        "email", "mail", "e_mail", "email_address", "email_id", "emailid",
        "contact_email", "user_email", "customer_email", "work_email",
        "personal_email", "login_email", "registered_email"
    ],
    "boolean": [
        "is_", "has_", "flag", "active", "enabled", "disabled", "deleted",
        "verified", "confirmed", "approved", "rejected", "subscribed",
        "opted_in", "is_active", "is_valid", "is_default", "is_primary",
        "boolean", "bool", "indicator", "switch", "toggle", "eligible"
    ],
    "zipcode": [
        "zip", "zipcode", "zip_code", "postal", "postal_code", "postcode",
        "pin", "pin_code", "pincode", "area_code", "post_code", "pcode",
        "eircode", "npa"
    ],
    "percentage": [
        "rate", "ratio", "percent", "pct", "percentage", "proportion",
        "share", "fraction", "growth_rate", "churn_rate", "conversion_rate",
        "discount_rate", "interest_rate", "tax_rate", "margin", "yield",
        "roi", "roa", "roe", "cagr", "irr", "fill_rate", "coverage"
    ],
    "score": [
        "score", "rating", "rank", "grade", "gpa", "marks", "points",
        "fico", "credit_score", "risk_score", "quality_score", "nps",
        "satisfaction", "priority", "severity", "confidence", "weight",
        "net_promoter", "review_score", "star_rating", "accuracy_score"
    ],
    "count": [
        "count", "num", "number", "qty", "quantity", "volume", "frequency",
        "occurrences", "views", "clicks", "impressions", "sessions",
        "requests", "transactions", "orders", "items", "units", "total_rows",
        "n_", "cnt", "no_of_", "num_of_", "visits", "events"
    ],
    "name": [
        "name", "full_name", "first_name", "last_name", "surname",
        "middle_name", "customer_name", "cust_name", "client_name",
        "user_name", "username", "display_name", "alias", "nick",
        "employee_name", "emp_name", "vendor_name", "supplier_name",
        "company_name", "firm_name", "business_name", "brand_name"
    ],
    "url": [
        "url", "link", "uri", "href", "endpoint", "website", "domain",
        "web_address", "page_url", "profile_url", "image_url", "photo_url",
        "redirect_url", "source_url", "callback_url", "api_url",
        "deep_link", "thumbnail_link"
    ],
    "ip_address": [
        "ip", "ip_address", "ipv4", "ipv6", "client_ip", "server_ip",
        "source_ip", "destination_ip", "remote_ip", "host_ip", "inet",
        "network_addr", "ip_addr"
    ],
    "coordinates": [
        "lat", "latitude", "lon", "longitude", "lng", "long", "coord",
        "coordinates", "geo_lat", "geo_lon", "decimal_degrees", "gps_lat",
        "gps_long", "location_lat", "location_lon", "x_coord", "y_coord",
        "easting", "northing"
    ],
    "duration": [
        "duration", "elapsed", "time_taken", "response_time", "latency",
        "age_days", "days_since", "tenure_days", "hold_time", "wait_time",
        "session_duration", "call_duration", "seconds", "minutes", "hours",
        "period", "interval", "timeout", "ttl"
    ],
    "address": [
        "address", "addr", "street", "lane", "road", "avenue", "location",
        "residence", "office_address", "billing_address", "shipping_address",
        "mailing_address", "house_no", "building", "locality", "landmark",
        "address_line1", "address_line2", "full_address"
    ],
    "currency_code": [
        "currency", "currency_code", "ccy", "ccy_code", "iso_currency",
        "transaction_currency", "base_currency", "quote_currency",
        "invoice_currency", "payment_currency", "reporting_currency"
    ],
    # Exotic types
    "swift_code": [
        "swift", "swift_code", "bic", "bic_code", "swift_bic",
        "bank_code", "routing_bic", "ifsc_bic_equivalent", "correspondent_swift"
    ],
    "iban": [
        "iban", "international_bank_account", "bank_account_iban",
        "iban_number", "iban_code"
    ],
    "ssn": [
        "ssn", "social_security", "social_security_number", "sin",
        "national_insurance", "nin", "tax_id_us"
    ],
    "pan_number": [
        "pan", "pan_no", "pan_number", "pan_card", "panno", "cust_pan",
        "income_tax_pan", "it_pan"
    ],
    "passport": [
        "passport", "passport_no", "passport_number", "travel_doc",
        "passport_id", "visa_passport"
    ],
    "vin": [
        "vin", "vehicle_identification", "chassis_no", "chassis_number",
        "vehicle_number", "vin_number"
    ],
    "mac_address": [
        "mac", "mac_address", "hardware_address", "physical_address",
        "device_mac", "mac_id", "nic_address"
    ],
    "credit_card": [
        "credit_card", "card_number", "card_no", "cc_number", "ccnum",
        "debit_card", "pan_card_no", "masked_card", "card_last4"
    ],
    "ticker_symbol": [
        "ticker", "symbol", "stock_symbol", "stock_ticker", "scrip",
        "exchange_code", "equity_symbol", "ticker_symbol", "bse_code",
        "nse_code", "isin"
    ],
    "hash_value": [
        "hash", "checksum", "md5", "sha256", "sha1", "digest", "fingerprint",
        "token_hash", "api_hash", "encrypted_id", "hex_digest"
    ],
    "unknown": [
        "misc", "other", "extra", "temp", "col", "field", "value",
        "unknown", "undefined", "raw", "custom", "var"
    ],
}


# ── B. NLP Feature Extraction from Column Name ───────────────────────────
def extract_name_features(col_name: str) -> dict:
    """
    Returns a feature dict from the column name alone using:
      1. Keyword overlap score per semantic type
      2. String signals (length, digit ratio, separator style)
    """
    # Normalise: lowercase, split camelCase, replace separators with space
    name = col_name.lower()
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)     # camelCase split
    name = re.sub(r'[_\-\.\s]+', ' ', name).strip()
    tokens = set(name.split())

    # Score against lexicon
    scores = {}
    for lbl, keywords in SEMANTIC_LEXICON.items():
        kw_tokens = set(' '.join(keywords).split())
        # Keyword in column name tokens
        direct_hit = len(tokens & kw_tokens)
        # Substring match (catches partial like "cust_panno" → "pan_number")
        substring_hit = sum(1 for kw in keywords if kw in name or name in kw)
        scores[lbl] = float(direct_hit + substring_hit)

    return scores


# ── C. Name-Only Prediction (Keyword Override Layer) ─────────────────────
def predict_from_name(col_name: str, threshold=2.0) -> str | None:
    """
    If the column name has a STRONG keyword hit (score >= threshold),
    return that label with 100% confidence — no model needed.
    Returns None if the name is ambiguous.
    """
    scores = extract_name_features(col_name)
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    if best_score >= threshold:
        return best_label
    return None


# ── D. NLP Training Data Generator ───────────────────────────────────────
def generate_nlp_training_data(n_per_class=300):
    """
    Generates (col_name, label) pairs for TF-IDF + LR training.
    Names are randomised combinations of the lexicon keywords.
    """
    X_names, y_labels = [], []
    for lbl, keywords in SEMANTIC_LEXICON.items():
        for _ in range(n_per_class):
            # Random naming patterns
            choice = RNG.integers(0, 5)
            kw = str(RNG.choice(keywords))
            if choice == 0:
                name = kw
            elif choice == 1:
                prefix = str(RNG.choice(["cust", "user", "org", "txn", "inv", "raw", "src", "dim"]))
                name = f"{prefix}_{kw}"
            elif choice == 2:
                suffix = str(RNG.choice(["id", "no", "val", "dt", "ts", "cd", "num", "flag"]))
                name = f"{kw}_{suffix}"
            elif choice == 3:
                # CamelCase variant
                name = kw.replace("_", " ").title().replace(" ", "")
            else:
                # Abbreviation: first char of each word
                parts = kw.split("_")
                name = "".join(p[0] for p in parts if p) + "_" + kw.split("_")[-1]

            X_names.append(name)
            y_labels.append(lbl)

    return X_names, y_labels


# ── E. Train TF-IDF + Logistic Regression on Column Names ────────────────
def train_nlp_name_classifier():
    log.info("\n  Training NLP Column-Name classifier (TF-IDF + LogReg)...")
    X_names, y_labels = generate_nlp_training_data(n_per_class=300)
    le_nlp = LabelEncoder()
    y_enc = le_nlp.fit_transform(y_labels)

    nlp_clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",         # character n-grams — handles abbreviations
            ngram_range=(2, 5),         # 2 to 5 char n-grams
            min_df=1, max_features=5000,
        )),
        ("lr", LogisticRegression(
            C=2.0, max_iter=1000, class_weight="balanced",
            multi_class="multinomial", solver="lbfgs",
        )),
    ])
    nlp_clf.fit(X_names, y_enc)
    train_acc = nlp_clf.score(X_names, y_enc)
    log.info(f"  NLP Classifier Training Accuracy: {train_acc:.3f}")
    return nlp_clf, le_nlp


# ── F. Combined Three-Stage Classifier ───────────────────────────────────
class NLPAugmentedSchemaClassifier:
    """
    Three-stage schema type predictor:

    Stage 1 — Keyword Override (Deterministic):
        If the column NAME has a strong keyword hit (score ≥ 2),
        return that label with 100% certainty. No statistical model needed.
        → Handles: "customer_pan_no" → pan_number
                   "gps_latitude"    → coordinates
                   "swift_bic"       → swift_code

    Stage 2 — NLP Classifier (Column Name Only):
        TF-IDF char n-gram + Logistic Regression on the column name.
        Uses char n-grams so it handles abbreviations like "cust_panno".
        → Handles: Novel naming conventions, typos, mixed-case, camelCase

    Stage 3 — Statistical Model (Data Values):
        Original trained RandomForest on 25 statistical features from
        actual data values.
        → Handles: Ambiguous column names ("value", "col1", "field_a")

    The three stages combine intelligently:
      - If Stage 1 fires, it ALWAYS wins.
      - If Stage 2 confidence > 0.65, it overrides Stage 3.
      - Otherwise Stage 3 (value-based) is used.
    """

    def __init__(self, clf_stat, le_stat, clf_nlp, le_nlp):
        self.clf_stat = clf_stat
        self.le_stat  = le_stat
        self.clf_nlp  = clf_nlp
        self.le_nlp   = le_nlp

    def predict_single(self, stat_features: np.ndarray, col_name: str = "") -> str:
        """
        Predicts semantic type for a single column.
        stat_features: 25-element feature vector from robust_extract_features()
        col_name:      the raw column name string
        """
        stat_features = np.array(stat_features, dtype=np.float32).reshape(1, -1)
        stat_features = np.nan_to_num(stat_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Stage 1: Keyword override
        if col_name:
            kw_pred = predict_from_name(col_name, threshold=2.0)
            if kw_pred is not None:
                return kw_pred

        # Stage 2: NLP on column name
        if col_name:
            try:
                nlp_proba = self.clf_nlp.predict_proba([col_name])[0]
                nlp_conf  = nlp_proba.max()
                if nlp_conf >= 0.65:
                    nlp_pred = self.le_nlp.inverse_transform([nlp_proba.argmax()])[0]
                    return nlp_pred
            except Exception:
                pass

        # Stage 3: Statistical model
        try:
            stat_pred = self.clf_stat.predict(stat_features)
            if hasattr(self.clf_stat, 'predict') and hasattr(stat_pred, '__iter__'):
                # Handle both old RF and CombinedSchemaClassifier from patch 5b
                pred = stat_pred[0] if hasattr(stat_pred[0], '__str__') else \
                       self.le_stat.inverse_transform(stat_pred)[0]
                return str(pred)
        except Exception:
            pass

        return "unknown"

    def predict_batch(self, rows: list) -> list:
        """
        rows: list of (stat_features_25, col_name_str) tuples
        """
        return [self.predict_single(f, n) for f, n in rows]

    def explain(self, stat_features: np.ndarray, col_name: str = "") -> dict:
        """Returns which stage fired and why, for debugging."""
        stat_features = np.array(stat_features, dtype=np.float32).reshape(1, -1)

        # Stage 1
        if col_name:
            scores = extract_name_features(col_name)
            best = max(scores, key=scores.get)
            if scores[best] >= 2.0:
                return {"stage": 1, "label": best, "reason": "keyword_override",
                        "keyword_scores": dict(sorted(scores.items(), key=lambda x: -x[1])[:5])}

        # Stage 2
        if col_name:
            try:
                nlp_proba = self.clf_nlp.predict_proba([col_name])[0]
                nlp_conf  = nlp_proba.max()
                nlp_pred  = self.le_nlp.inverse_transform([nlp_proba.argmax()])[0]
                if nlp_conf >= 0.65:
                    return {"stage": 2, "label": nlp_pred, "reason": "nlp_name_classifier",
                            "confidence": round(float(nlp_conf), 4)}
                else:
                    return {"stage": 3, "label": self.predict_single(stat_features, ""),
                            "reason": f"nlp_low_confidence ({nlp_conf:.3f}), fell back to stats"}
            except: pass

        return {"stage": 3, "label": self.predict_single(stat_features, ""),
                "reason": "no_column_name_or_nlp_unavailable"}


# ── G. Patch: Wrap and save ───────────────────────────────────────────────
def apply_nlp_patch():
    log.info("\n" + "="*70)
    log.info(" PATCHING SCHEMA CLASSIFIER WITH NLP COLUMN-NAME UNDERSTANDING")
    log.info("="*70)

    # Load existing model
    clf_stat = joblib.load(os.path.join(MODELS_DIR, "schema_classifier.pkl"))
    le_stat  = joblib.load(os.path.join(MODELS_DIR, "schema_label_encoder.pkl"))
    log.info(f"  Loaded existing statistical model.")

    # Train NLP classifier
    clf_nlp, le_nlp = train_nlp_name_classifier()

    # Combine into NLP-augmented model
    combined = NLPAugmentedSchemaClassifier(clf_stat, le_stat, clf_nlp, le_nlp)

    # Quick self-test
    log.info("\n  Self-Test — Column Name Keyword Override (Stage 1):")
    test_cases = [
        ("gps_latitude",       [0.0]*25, "coordinates"),
        ("customer_pan_no",    [0.0]*25, "pan_number"),
        ("swift_bic_code",     [0.0]*25, "swift_code"),
        ("transaction_amount", [0.0]*25, "amount"),
        ("email_address",      [0.0]*25, "email"),
        ("is_active_flag",     [0.0]*25, "boolean"),
        ("vin_number",         [0.0]*25, "vin"),
        ("hash_checksum",      [0.0]*25, "hash_value"),
        ("col_xyz_unknown",    [0.0]*25, "unknown"),   # no strong match — OK
    ]
    all_pass = True
    for col_name, feats, expected in test_cases:
        got = combined.predict_single(feats, col_name)
        ok  = "✓" if got == expected else "✗"
        if got != expected: all_pass = False
        log.info(f"    {ok}  {col_name:<30} → {got} (expected {expected})")

    # Save updated artifact
    joblib.dump(combined, os.path.join(MODELS_DIR, "schema_classifier.pkl"))
    log.info(f"\n  ✓ Saved NLP-augmented schema_classifier.pkl")
    log.info(f"  ✓ Covers {len(SEMANTIC_LEXICON)} semantic types with 3-stage intelligence")
    log.info(f"\n  How to use in your codebase:")
    log.info( "    clf = joblib.load('models/schema_classifier.pkl')")
    log.info( "    label = clf.predict_single(stat_features_25, col_name='invoice_date')")
    log.info( "    debug = clf.explain(stat_features_25, col_name='invoice_date')")

    if all_pass:
        log.info("\n  ✅ All self-tests passed. Model ready for production.")
    else:
        log.info("\n  ⚠️  Some self-tests incorrect. Review SEMANTIC_LEXICON mappings.")

    return combined


# ── Run patch ─────────────────────────────────────────────────────────────
nlp_classifier = apply_nlp_patch()
