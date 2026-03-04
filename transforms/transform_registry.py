"""
transforms/transform_registry.py
----------------------------------
Central registry for all data transformation functions in DIPEX.

The TransformRegistry acts as a catalogue of named, versioned transform
functions that can be composed into processing pipelines. Transforms are
registered by name and can be retrieved, listed, and applied to DataFrames.

Typical usage
-------------
    registry = TransformRegistry()
    registry.register("normalize", normalize_fn, version="1.0")
    fn = registry.get("normalize")
    df_out = fn(df_in)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("dipex.transforms.registry")


class TransformRegistry:
    """
    A thread-safe registry for named data transform functions.

    Features
    --------
    - Register transforms with name, version, tags, and description
    - Retrieve transforms by name (latest version or specific version)
    - List all registered transforms with metadata
    - Apply a named transform directly to a DataFrame
    - Chain multiple named transforms in sequence
    """

    def __init__(self) -> None:
        # { name: List[(version, callable, metadata)] }
        self._registry: Dict[str, List[Tuple[str, Callable, Dict[str, Any]]]] = {}

    def register(
        self,
        name: str,
        fn: Callable[[pd.DataFrame], pd.DataFrame],
        version: str = "1.0",
        description: str = "",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a transform function under the given name and version.

        Parameters
        ----------
        name : str
            Unique name for this transform.
        fn : Callable[[pd.DataFrame], pd.DataFrame]
            Transform function: receives a DataFrame, returns a DataFrame.
        version : str
            Semver string, e.g. "1.0", "2.1.3".
        description : str
            Human-readable description of what this transform does.
        tags : list of str, optional
            Categorical tags for filtering (e.g. ["cleaning", "numeric"]).
        overwrite : bool
            If True, overwrite an existing version of the same name.

        Raises
        ------
        ValueError
            If the name+version combination already exists and overwrite=False.
        """
        if not callable(fn):
            raise TypeError(f"Transform '{name}' must be callable, got {type(fn)}")

        meta = {
            "version": version,
            "description": description,
            "tags": tags or [],
        }

        entries = self._registry.setdefault(name, [])
        # Check for duplicate version
        for i, (v, _, _) in enumerate(entries):
            if v == version:
                if overwrite:
                    entries[i] = (version, fn, meta)
                    logger.info("TransformRegistry: overwritten '%s' v%s", name, version)
                    return
                else:
                    raise ValueError(
                        f"Transform '{name}' v{version} already registered. "
                        f"Use overwrite=True to replace it."
                    )
        entries.append((version, fn, meta))
        logger.debug("TransformRegistry: registered '%s' v%s", name, version)

    def get(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """
        Retrieve a registered transform by name (and optionally version).

        Parameters
        ----------
        name : str
            Registered transform name.
        version : str, optional
            If None, returns the latest registered version.

        Returns
        -------
        Callable

        Raises
        ------
        KeyError
            If the name (or specific version) is not found.
        """
        if name not in self._registry:
            raise KeyError(
                f"Transform '{name}' not found. "
                f"Available: {list(self._registry)}"
            )
        entries = self._registry[name]
        if version is None:
            # Return the last registered (latest) version
            return entries[-1][1]
        for v, fn, _ in entries:
            if v == version:
                return fn
        raise KeyError(
            f"Transform '{name}' v{version} not found. "
            f"Available versions: {[e[0] for e in entries]}"
        )

    def apply(
        self,
        name: str,
        df: pd.DataFrame,
        version: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Apply a registered transform to a DataFrame.

        Parameters
        ----------
        name : str
            Registered transform name.
        df : pd.DataFrame
            Input DataFrame.
        version : str, optional
            Specific version to apply; defaults to latest.
        **kwargs
            Passed to the transform function if it accepts keyword args.

        Returns
        -------
        pd.DataFrame
        """
        fn = self.get(name, version=version)
        try:
            return fn(df, **kwargs) if kwargs else fn(df)
        except TypeError:
            # Function doesn't accept kwargs — call without them
            return fn(df)

    def chain(
        self,
        names: List[str],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply a sequence of named transforms in order.

        Parameters
        ----------
        names : list of str
            Transform names to apply in sequence.
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame after all transforms have been applied.
        """
        result = df
        for name in names:
            result = self.apply(name, result)
            logger.debug("TransformRegistry: applied '%s' → %d rows", name, len(result))
        return result

    def list_transforms(self) -> List[Dict[str, Any]]:
        """
        Return metadata for all registered transforms.

        Returns
        -------
        list of dict with keys: name, version, description, tags
        """
        out = []
        for name, entries in self._registry.items():
            for version, _, meta in entries:
                out.append({"name": name, **meta})
        return out

    def names(self) -> List[str]:
        """Return all registered transform names (unique)."""
        return list(self._registry.keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self._registry.values())

    def __repr__(self) -> str:
        return f"TransformRegistry({len(self)} transforms registered)"


# ── Module-level default registry (singleton for convenience) ─────────────────

_default_registry = TransformRegistry()


def register(
    name: str,
    fn: Callable[[pd.DataFrame], pd.DataFrame],
    version: str = "1.0",
    description: str = "",
    tags: Optional[List[str]] = None,
) -> None:
    """Register a transform in the module-level default registry."""
    _default_registry.register(name, fn, version=version, description=description, tags=tags)


def get(name: str, version: Optional[str] = None) -> Callable:
    """Get a transform from the module-level default registry."""
    return _default_registry.get(name, version=version)
