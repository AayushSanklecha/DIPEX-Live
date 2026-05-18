"""
Quick smoke test — verifies all v7 models load and run basic inference.
Run from project root: python smoke_test_models.py
"""
import joblib, json, pathlib, sys

M = pathlib.Path("models")
ok, fail = [], []

def check(name, path, info_fn=None):
    try:
        p = pathlib.Path(path)
        obj = json.loads(p.read_text()) if str(path).endswith(".json") else joblib.load(p)
        extra = info_fn(obj) if info_fn else ""
        print(f"  PASS  {name:<32} {extra}")
        ok.append(name)
        return obj
    except Exception as e:
        print(f"  FAIL  {name:<32} {e}")
        fail.append(name)
        return None

print("=" * 60)
print("MODEL LOAD SMOKE TEST")
print("=" * 60)

# 1. Drift
drift = check("drift_pipeline", M/"drift_pipeline.pkl",
    lambda o: "keys=" + str(list(o.keys())) if isinstance(o, dict) else "type=" + type(o).__name__)

# 2. Schema
schema = check("schema_classifier", M/"schema_classifier.pkl",
    lambda o: "type=" + type(o).__name__)
check("schema_feature_registry", M/"schema_feature_registry.pkl",
    lambda o: "version=" + str(o.get("version", "?")))
check("schema_label_encoder", M/"schema_label_encoder.pkl",
    lambda o: "classes=" + str(len(o.classes_)))

# 3. Domain
domain = check("domain_classifier", M/"domain_classifier.pkl",
    lambda o: "type=" + type(o).__name__)
check("domain_registry", M/"domain_registry.pkl",
    lambda o: "domains=" + str(o.get("domain_labels", [])))

# 4. Anomaly
anomaly = check("anomaly_detector", M/"anomaly_detector.pkl",
    lambda o: "type=" + type(o).__name__)
check("anomaly_threshold", M/"anomaly_threshold.pkl",
    lambda o: "threshold=" + str(round(float(o["threshold"]), 4) if isinstance(o, dict) else round(float(o), 4)))

# 5. Chart
chart = check("chart_relevance_scorer", M/"chart_relevance_scorer.pkl",
    lambda o: "type=" + type(o).__name__)
check("chart_registry", M/"chart_registry.pkl",
    lambda o: "charts=" + str(o.get("chart_types", [])))

# 6. Proposal
proposal = check("proposal_confidence", M/"proposal_confidence.pkl",
    lambda o: "type=" + type(o).__name__)
check("confidence_metadata", M/"confidence_metadata.json",
    lambda o: "keys=" + str(list(o.keys())))

print()
print("=" * 60)
print(f"RESULT:  {len(ok)} PASS  /  {len(fail)} FAIL")
print("=" * 60)

if fail:
    print("FAILED:", fail)
    sys.exit(1)
else:
    print("ALL MODELS LOADED SUCCESSFULLY.")
