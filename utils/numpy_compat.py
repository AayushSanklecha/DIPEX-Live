# utils/numpy_compat.py
"""
NumPy 2.x compatibility guard.
Run this on startup. Catches any remaining legacy alias usage before
it silently corrupts array operations.

Issue 02: NumPy 2.0 removed all legacy type aliases that existed since 1.x.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

NUMPY_REMOVED_ALIASES = [
    "int", "float", "complex", "object", "str", "unicode",
]


def check_numpy_compatibility() -> None:
    """
    Validate that the installed NumPy version is compatible with DIPEX v3.

    - If NumPy < 2: warn and recommend upgrade
    - If NumPy 2.x still exposes removed aliases: raise (corrupted install)
    - If clean: log success
    """
    major = int(np.__version__.split(".")[0])
    if major < 2:
        logger.warning(
            "NumPy version %s detected. DIPEX v3 requires NumPy 2.x. "
            "Upgrade with: pip install 'numpy>=2.1.3,<2.2'",
            np.__version__,
        )
        return

    broken: list[str] = []
    for alias in NUMPY_REMOVED_ALIASES:
        if alias in dir(np):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    obj = getattr(np, alias)
            except AttributeError:
                continue
            # In NumPy 2.x some names exist but as builtins (e.g. np.bool is bool).
            # Only flag if they are NOT the Python builtin.
            builtin_map = {
                "bool": bool, "int": int, "float": float,
                "complex": complex, "object": object, "str": str,
            }
            if alias in builtin_map and obj is builtin_map[alias]:
                continue  # This is fine — NumPy re-exports the builtin
            broken.append(f"np.{alias}")

    if broken:
        raise RuntimeError(
            f"NumPy {np.__version__} still exposes removed aliases: {broken}. "
            "This indicates a corrupted or patched NumPy install. "
            "Run: pip install --force-reinstall numpy==2.1.3"
        )

    logger.info("NumPy %s compatibility check passed.", np.__version__)
