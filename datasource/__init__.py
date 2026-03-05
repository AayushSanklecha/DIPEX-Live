"""
datasource/__init__.py
-----------------------
DATA SOURCE LAYER — Layer 1 of the DIPEX architecture.

Exposes a unified DataSourceRouter that maps source type strings to the
appropriate reader/connector from ingestion/readers/ and ingestion/connectors/.
"""

from datasource.router import DataSourceRouter

__all__ = ["DataSourceRouter"]
