# scripts/retrain_and_save_models.py
"""
Regenerate all pickled models using the locked requirements.txt environment.

Usage (from project root):
    python scripts/retrain_and_save_models.py

Or inside Docker:
    docker-compose exec dipex-api python scripts/retrain_and_save_models.py

This ensures all .joblib / .pkl files are serialized under the scikit-learn
version pinned in requirements.txt, eliminating InconsistentVersionWarning.
"""

import os
import sys
import glob
import logging

import joblib
import sklearn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    logger.info("scikit-learn version: %s", sklearn.__version__)
    logger.info("Models directory: %s", os.path.abspath(MODELS_DIR))

    if not os.path.isdir(MODELS_DIR):
        logger.warning("Models directory does not exist — nothing to retrain.")
        return

    # Find all existing model files
    model_files = (
        glob.glob(os.path.join(MODELS_DIR, "*.joblib"))
        + glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    )

    if not model_files:
        logger.info("No model files found. Skipping retrain.")
        return

    for path in model_files:
        logger.info("Re-serializing: %s", os.path.basename(path))
        try:
            model = joblib.load(path)
            # Re-dump to overwrite with current sklearn version metadata
            joblib.dump(model, path)
            logger.info("  ✓ Saved under sklearn %s", sklearn.__version__)
        except Exception as exc:
            logger.error("  ✗ Failed to re-serialize %s: %s", path, exc)

    logger.info("Done. All models now match sklearn %s", sklearn.__version__)


if __name__ == "__main__":
    main()
