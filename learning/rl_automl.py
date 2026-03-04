"""
learning/rl_automl.py
---------------------
DEPRECATED SHIM — do not import this module directly.

The canonical RL AutoML agent (Q-learning pipeline selector) lives in:

    modeling.rl_automl

This file exists only to preserve backward compatibility for any code
that still imports from learning.rl_automl.  All symbols are re-exported
from modeling.rl_automl transparently.

Migration
---------
Replace:
    from learning.rl_automl import RLAutoML
With:
    from modeling.rl_automl import RLAutoML
"""

import warnings

warnings.warn(
    "learning.rl_automl is deprecated and will be removed in a future release. "
    "Use 'from modeling.rl_automl import RLAutoML' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical module so existing imports continue to work
from modeling.rl_automl import *          # noqa: F401, F403, E402
from modeling.rl_automl import RLAutoML, get_rl_automl, ALL_ACTIONS  # noqa: E402

__all__ = ["RLAutoML", "get_rl_automl", "ALL_ACTIONS"]
