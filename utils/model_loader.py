# utils/model_loader.py
"""
Safe model loader with scikit-learn version validation.

Raises RuntimeError if a loaded model was trained on a different sklearn
version, forcing the user to retrain via scripts/retrain_and_save_models.py.
"""

import logging
import warnings

import joblib
import sklearn
import sklearn.exceptions

logger = logging.getLogger(__name__)


def load_model_safe(path: str):
    """
    Load a joblib/pickle model with sklearn version guard.

    If the model was pickled under a different sklearn version, a RuntimeError
    is raised instead of silently returning potentially corrupt predictions.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(path)
        for w in caught:
            if issubclass(w.category, sklearn.exceptions.InconsistentVersionWarning):
                raise RuntimeError(
                    f"Model at {path} was trained on a different sklearn version. "
                    f"Current: {sklearn.__version__}. "
                    f"Re-run: docker-compose exec dipex-api python scripts/retrain_and_save_models.py"
                )
    logger.debug("Loaded model %s (sklearn %s)", path, sklearn.__version__)
    return model
