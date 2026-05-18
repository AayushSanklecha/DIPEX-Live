#!/usr/bin/env python3
"""
================================================================
 ADAP v7 — POST-TRAINING VALIDATION (Run after all 6 models)
================================================================
HOW TO RUN IN COLAB:
  1. Run Cell 0 (pip install)
  2. Paste & run 00_shared_utils.py as Cell 1
  3. After ALL 6 model scripts complete, paste & run THIS file

HOW TO RUN STANDALONE (local / after running 00→06):
  python 07_post_validation.py

Verifies all 6 model .pkl files exist, load cleanly, and have
consistent NLP method metadata.
================================================================
"""

from datetime import datetime as _datetime

# ── Standalone bootstrap ──────────────────────────────────────────────────────
# When run as a plain Python script (not inside a Colab notebook that already
# executed 00_shared_utils.py), we load the shared module dynamically so that
# all globals (log, MODELS_DIR, NLP, joblib, json, Path, VERSION …) are present.
import importlib.util as _ilu, sys as _sys, pathlib as _pl
if "log" not in dir():          # guard: skip if already injected by Colab cell
    # __file__ is undefined in Colab notebooks — use cwd fallback
    try:
        _HERE = _pl.Path(__file__).resolve().parent
    except NameError:
        _HERE = _pl.Path.cwd()   # Colab: look in the current working directory
    _spec = _ilu.spec_from_file_location("shared_utils", _HERE / "00_shared_utils.py")
    if _spec is None or not (_HERE / "00_shared_utils.py").exists():
        raise RuntimeError(
            "Cannot find 00_shared_utils.py. In Colab, make sure you ran "
            "Cell 1 (00_shared_utils.py) before running this cell."
        )
    _su   = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_su)            # type: ignore[union-attr]
    globals().update(vars(_su))             # inject all shared globals here
# ─────────────────────────────────────────────────────────────────────────────


def run_post_training_validation():
    log.info("\n=== POST-TRAINING VALIDATION (All 6 Models) ===")
    models_to_check = [
        ("drift_autoencoder",      f"{MODELS_DIR}/drift_pipeline.pkl",          None),
        ("schema_classifier",      f"{MODELS_DIR}/schema_classifier.pkl",        f"{MODELS_DIR}/schema_feature_registry.pkl"),
        ("domain_classifier",      f"{MODELS_DIR}/domain_classifier.pkl",        f"{MODELS_DIR}/domain_registry.pkl"),
        ("anomaly_detector",       f"{MODELS_DIR}/anomaly_detector.pkl",         f"{MODELS_DIR}/anomaly_threshold.pkl"),
        ("chart_relevance_scorer", f"{MODELS_DIR}/chart_relevance_scorer.pkl",   f"{MODELS_DIR}/chart_registry.pkl"),
        ("proposal_confidence",    f"{MODELS_DIR}/proposal_confidence.pkl",      f"{MODELS_DIR}/confidence_metadata.json"),
    ]

    results = {}
    all_pass = True
    for name, model_path, meta_path in models_to_check:
        result = {"name": name, "model_exists": Path(model_path).exists(), "checks": []}

        if not result["model_exists"]:
            result["checks"].append({"check": "file_exists", "passed": False,
                                     "detail": f"MISSING: {model_path}"})
            all_pass = False
        else:
            sz_mb = Path(model_path).stat().st_size / 1e6
            result["size_mb"] = round(sz_mb, 2)
            result["checks"].append({"check": "file_exists", "passed": True,
                                     "detail": f"{sz_mb:.1f} MB"})
            try:
                artifact = joblib.load(model_path)
                result["checks"].append({"check": "loadable", "passed": True})
                # Extra check for drift AE: verify PyTorch dict format (not old sklearn Pipeline)
                if name == "drift_autoencoder":
                    is_torch_format = (isinstance(artifact, dict)
                                       and artifact.get("type") == "torch_ae_v7")
                    result["checks"].append({
                        "check": "drift_ae_torch_format",
                        "passed": is_torch_format,
                        "detail": ("torch_ae_v7 dict ✅" if is_torch_format
                                   else f"unexpected format: {type(artifact).__name__} — "
                                        "did you run old monolithic script instead of "
                                        "01_drift_autoencoder.py?"),
                    })
            except Exception as e:
                result["checks"].append({"check": "loadable", "passed": False,
                                         "detail": str(e)[:100]})
                all_pass = False

        if meta_path and Path(str(meta_path)).exists():
            try:
                if str(meta_path).endswith(".json"):
                    with open(meta_path) as f:
                        meta = json.load(f)
                else:
                    meta = joblib.load(meta_path)
                result["checks"].append({"check": "metadata_exists", "passed": True})
                if "nlp_method" in meta:
                    nlp_match = (meta["nlp_method"] == NLP._method)
                    result["checks"].append({
                        "check": "nlp_method_consistent",
                        "passed": nlp_match,
                        "detail": f"saved={meta['nlp_method']} current={NLP._method}",
                    })
                    if not nlp_match:
                        log.warning("  ⚠️  [%s] NLP mismatch: trained=%s, now=%s",
                                    name, meta["nlp_method"], NLP._method)
                        all_pass = False
                if "version" in meta:
                    result["version"] = meta["version"]
            except Exception as e:
                result["checks"].append({"check": "metadata_load", "passed": False,
                                         "detail": str(e)[:80]})

        passed = all(c["passed"] for c in result["checks"])
        result["passed"] = passed
        if not passed:
            all_pass = False
        results[name] = result
        status = "✅" if passed else "❌"
        log.info("  %s %-30s  %s MB", status, name,
                 f"{result.get('size_mb', '?')}")

    val_report = {
        "timestamp": _datetime.utcnow().isoformat(),
        "version": VERSION,
        "all_passed": all_pass,
        "models": results,
    }
    report_path = f"{REPORTS_DIR}/post_training_validation.json"
    with open(report_path, "w") as f:
        json.dump(val_report, f, indent=2, default=str)

    if all_pass:
        log.info("\n  ✅ ALL 6 MODELS PASSED POST-TRAINING VALIDATION")
        log.info("  Download from: %s", MODELS_DIR)
        log.info("  Copy to:       dipex_project/models/")
    else:
        log.warning("\n  ⚠️  SOME MODELS FAILED — see %s", report_path)

    # Summary table
    print("\n" + "="*60)
    print(f"{'Model':<30} {'Status':<8} {'Size MB'}")
    print("-"*60)
    for name, r in results.items():
        status  = "PASS" if r.get("passed") else "FAIL"
        size_mb = r.get("size_mb", "?")
        print(f"{name:<30} {status:<8} {size_mb}")
    print("="*60)
    print(f"Overall: {'ALL PASS ✅' if all_pass else 'FAILURES DETECTED ❌'}")
    return val_report


if __name__ == "__main__":
    run_post_training_validation()

    # Show disk usage
    total_mb = 0.0
    print("\nModel artifacts:")
    for f in sorted(Path(MODELS_DIR).iterdir()):
        if f.is_file():
            mb = f.stat().st_size / 1e6
            total_mb += mb
            print(f"  {f.name:<55}  {mb:6.2f} MB")
    print(f"\n  TOTAL: {total_mb:.1f} MB")
    print(f"  Source:  {MODELS_DIR}")
    print(f"  Copy to: dipex_project/models/")
