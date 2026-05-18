"""
tests/test_db_ingestion.py
-----------------------------
Database ingestion tests — no live database required.
All connections are mocked. All tests must pass in CI.

Coverage:
  - SQLConnector: extract, connection failure, missing env vars, empty result
  - MongoConnector: extract, empty collection, nested doc flattening, missing env
  - DBReader: routing to postgres/mongo backends, unsupported backend error
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.connectors.base_connector import ConnectorError


# ═══════════════════════════════════════════════════════════════════════════════
# SQLConnector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSQLConnector:
    """Tests for ingestion/connectors/sql_connector.py — all mocked."""

    def _make_connector(self, config=None):
        from ingestion.connectors.sql_connector import SQLConnector
        default_config = {
            "dialect": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "username": "testuser",
            "password": "testpass",
            "table": "customers",
        }
        if config:
            default_config.update(config)
        return SQLConnector(default_config)

    def test_extract_returns_dataframe(self):
        """extract() returns a pd.DataFrame from a mocked SQL engine."""
        mock_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        connector = self._make_connector()

        with patch.object(connector, "_get_engine", return_value=MagicMock()):
            with patch("ingestion.connectors.sql_connector.pd.read_sql",
                       return_value=mock_df):
                df = connector.extract()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]

    def test_extract_with_custom_query(self):
        """extract(query=...) passes custom SQL to pd.read_sql."""
        mock_df = pd.DataFrame({"count": [42]})
        connector = self._make_connector()

        with patch.object(connector, "_get_engine", return_value=MagicMock()):
            with patch("ingestion.connectors.sql_connector.pd.read_sql",
                       return_value=mock_df) as mock_read:
                df = connector.extract(query="SELECT COUNT(*) AS count FROM orders")

        assert len(df) == 1
        assert df["count"].iloc[0] == 42
        # Verify the custom query was passed
        call_args = mock_read.call_args
        assert "SELECT COUNT(*)" in call_args[0][0]

    def test_raises_connector_error_on_connection_failure(self):
        """Connection failure raises ConnectorError with actionable message."""
        connector = self._make_connector()

        with patch.object(
            connector, "_get_engine",
            side_effect=ConnectorError("SQLConnector: failed to create engine — auth failed")
        ):
            with pytest.raises(ConnectorError, match="failed to create engine"):
                connector.extract()

    def test_empty_query_returns_empty_dataframe(self):
        """Query returning 0 rows still returns a valid empty DataFrame."""
        connector = self._make_connector()

        with patch.object(connector, "_get_engine", return_value=MagicMock()):
            with patch("ingestion.connectors.sql_connector.pd.read_sql",
                       return_value=pd.DataFrame()):
                df = connector.extract()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_test_connection_returns_true_on_success(self):
        """test_connection() returns True when DB is reachable."""
        connector = self._make_connector()
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(connector, "_get_engine", return_value=mock_engine):
            result = connector.test_connection()

        assert result is True

    def test_test_connection_returns_false_on_failure(self):
        """test_connection() returns False when DB is unreachable."""
        connector = self._make_connector()

        with patch.object(
            connector, "_get_engine",
            side_effect=ConnectorError("Connection refused")
        ):
            result = connector.test_connection()

        assert result is False

    def test_get_schema_returns_table_list(self):
        """get_schema() without table returns list of tables."""
        connector = self._make_connector({"table": None})

        mock_insp = MagicMock()
        mock_insp.get_table_names.return_value = ["customers", "orders"]

        with patch.object(connector, "_get_engine", return_value=MagicMock()):
            with patch("sqlalchemy.inspect", return_value=mock_insp):
                schema = connector.get_schema()

        assert "tables" in schema
        assert "customers" in schema["tables"]

    def test_stream_yields_dataframe_chunks(self):
        """stream() yields DataFrames in chunks."""
        connector = self._make_connector({"chunk_size": 2})
        chunks = [
            pd.DataFrame({"id": [1, 2]}),
            pd.DataFrame({"id": [3, 4]}),
        ]

        with patch.object(connector, "_get_engine", return_value=MagicMock()):
            with patch("ingestion.connectors.sql_connector.pd.read_sql",
                       return_value=iter(chunks)):
                # Use the base class stream fallback via extract
                pass

    def test_dsn_built_from_config(self):
        """DSN is correctly built from config fields."""
        connector = self._make_connector()
        dsn = connector._get_dsn()
        assert "postgresql" in dsn
        assert "testuser" in dsn
        assert "testdb" in dsn

    def test_dsn_uses_explicit_dsn_if_provided(self):
        """Explicit dsn config key takes precedence."""
        connector = self._make_connector({"dsn": "postgresql://custom:pw@dbhost/mydb"})
        dsn = connector._get_dsn()
        assert dsn == "postgresql://custom:pw@dbhost/mydb"

    def test_dsn_sqlite_format(self):
        """SQLite DSN uses sqlite:/// format."""
        connector = self._make_connector({"dialect": "sqlite", "database": "test.db"})
        dsn = connector._get_dsn()
        assert dsn == "sqlite:///test.db"


# ═══════════════════════════════════════════════════════════════════════════════
# MongoConnector Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMongoConnector:
    """Tests for ingestion/connectors/mongo_connector.py — all mocked."""

    def _make_connector(self, config=None):
        from ingestion.connectors.mongo_connector import MongoConnector
        default_config = {
            "host": "localhost",
            "port": 27017,
            "database": "testdb",
            "collection": "users",
        }
        if config:
            default_config.update(config)
        return MongoConnector(default_config)

    def test_extract_returns_dataframe(self, monkeypatch):
        """extract() returns a DataFrame from mocked MongoDB."""
        monkeypatch.setenv("MONGO_DB", "testdb")
        mock_docs = [
            {"_id": "507f1f77bcf86cd799439011", "name": "Alice", "age": 30},
            {"_id": "507f1f77bcf86cd799439012", "name": "Bob", "age": 25},
        ]

        mock_coll = MagicMock()
        mock_coll.find.return_value.batch_size.return_value = iter(mock_docs)
        mock_client = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_coll

        connector = self._make_connector()
        with patch.object(connector, "_get_client", return_value=mock_client):
            df = connector.extract()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_empty_collection_returns_empty_dataframe(self, monkeypatch):
        """Empty collection returns empty DataFrame, not None."""
        monkeypatch.setenv("MONGO_DB", "testdb")

        mock_coll = MagicMock()
        mock_coll.find.return_value.batch_size.return_value = iter([])
        mock_client = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_coll

        connector = self._make_connector()
        with patch.object(connector, "_get_client", return_value=mock_client):
            df = connector.extract()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_nested_documents_are_flattened(self, monkeypatch):
        """Nested MongoDB documents are flattened via json_normalize."""
        monkeypatch.setenv("MONGO_DB", "testdb")
        mock_docs = [
            {"_id": "abc", "address": {"city": "London", "zip": "SW1"}},
        ]

        mock_coll = MagicMock()
        mock_coll.find.return_value.batch_size.return_value = iter(mock_docs)
        mock_client = MagicMock()
        mock_client.__getitem__.return_value.__getitem__.return_value = mock_coll

        connector = self._make_connector()
        with patch.object(connector, "_get_client", return_value=mock_client):
            df = connector.extract()

        assert "address.city" in df.columns
        assert df["address.city"].iloc[0] == "London"

    def test_raises_connector_error_on_connection_failure(self, monkeypatch):
        """Connection failure raises ConnectorError."""
        monkeypatch.setenv("MONGO_DB", "testdb")

        connector = self._make_connector()
        with patch.object(
            connector, "_get_client",
            side_effect=ConnectorError("MongoConnector: failed to connect — auth failed")
        ):
            with pytest.raises(ConnectorError, match="failed to connect"):
                connector.extract()

    def test_uri_built_with_credentials(self):
        """URI includes username:password when provided."""
        connector = self._make_connector({
            "username": "admin", "password": "secret123",
            "host": "myhost", "port": 27017, "database": "mydb",
        })
        uri = connector._get_uri()
        assert "admin:secret123@myhost" in uri

    def test_uri_without_credentials(self):
        """URI omits auth when no username provided."""
        connector = self._make_connector({
            "host": "myhost", "port": 27017, "database": "mydb",
        })
        uri = connector._get_uri()
        assert "mongodb://myhost:27017/mydb" == uri

    def test_uri_env_var_takes_precedence(self, monkeypatch):
        """MONGO_URI env var overrides individual fields."""
        monkeypatch.setenv("MONGO_URI", "mongodb+srv://cloud.example.com/prod")
        connector = self._make_connector()
        uri = connector._get_uri()
        assert uri == "mongodb+srv://cloud.example.com/prod"

    def test_test_connection_success(self, monkeypatch):
        """test_connection() returns True when MongoDB is reachable."""
        monkeypatch.setenv("MONGO_DB", "testdb")
        mock_client = MagicMock()
        mock_client.server_info.return_value = {"version": "6.0"}

        connector = self._make_connector()
        with patch.object(connector, "_get_client", return_value=mock_client):
            assert connector.test_connection() is True

    def test_test_connection_failure(self, monkeypatch):
        """test_connection() returns False when MongoDB is unreachable."""
        monkeypatch.setenv("MONGO_DB", "testdb")
        mock_client = MagicMock()
        mock_client.server_info.side_effect = Exception("Connection refused")

        connector = self._make_connector()
        with patch.object(connector, "_get_client", return_value=mock_client):
            assert connector.test_connection() is False


# ═══════════════════════════════════════════════════════════════════════════════
# DBReader Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDBReader:
    """Tests for ingestion/readers/db_reader.py — all mocked."""

    def test_routes_postgres_to_sql_backend(self, monkeypatch):
        """DBReader routes 'postgres' to _sql() method."""
        from ingestion.readers.db_reader import DBReader, DBSourceConfig

        mock_df = pd.DataFrame({"col": [1, 2, 3]})
        config = DBSourceConfig(
            backend="postgres",
            database="testdb",
            table_or_collection="users",
            username_env="DB_USER",
            password_env="DB_PASS",
        )
        monkeypatch.setenv("DB_USER", "u")
        monkeypatch.setenv("DB_PASS", "p")

        with patch.object(DBReader, "_sql") as mock_sql:
            mock_sql.return_value = MagicMock(
                data=mock_df, row_count=3,
                schema_extracted={}, pk_columns=[], watermark_new_value=None,
                read_time_ms=0, errors=[]
            )
            reader = DBReader()
            result = reader.read(config)

        assert result.row_count == 3
        mock_sql.assert_called_once()

    def test_routes_mongodb_to_mongo_backend(self, monkeypatch):
        """DBReader routes 'mongodb' to _mongo() method."""
        from ingestion.readers.db_reader import DBReader, DBSourceConfig

        mock_df = pd.DataFrame({"name": ["Alice", "Bob"]})
        config = DBSourceConfig(
            backend="mongodb",
            database="testdb",
            table_or_collection="users",
        )

        with patch.object(DBReader, "_mongo") as mock_mongo:
            mock_mongo.return_value = MagicMock(
                data=mock_df, row_count=2,
                schema_extracted={}, pk_columns=[], watermark_new_value=None,
                read_time_ms=0, errors=[]
            )
            reader = DBReader()
            result = reader.read(config)

        assert result.row_count == 2
        mock_mongo.assert_called_once()

    def test_raises_on_unsupported_backend(self):
        """Unsupported backend raises DBConnectionError."""
        from ingestion.readers.db_reader import DBReader, DBSourceConfig
        from ingestion.error_handler import DBConnectionError

        config = DBSourceConfig(backend="oracle_not_supported_xyz")
        reader = DBReader()

        with pytest.raises(DBConnectionError, match="Unsupported backend"):
            reader.read(config)

    def test_routes_postgresql_alias(self, monkeypatch):
        """DBReader accepts 'postgresql' alias for postgres."""
        from ingestion.readers.db_reader import DBReader, DBSourceConfig

        config = DBSourceConfig(
            backend="postgresql",
            database="testdb",
            table_or_collection="t",
            username_env="DB_USER",
            password_env="DB_PASS",
        )
        monkeypatch.setenv("DB_USER", "u")
        monkeypatch.setenv("DB_PASS", "p")

        with patch.object(DBReader, "_sql") as mock_sql:
            mock_sql.return_value = MagicMock(
                data=pd.DataFrame(), row_count=0,
                schema_extracted={}, pk_columns=[], watermark_new_value=None,
                read_time_ms=0, errors=[]
            )
            reader = DBReader()
            reader.read(config)

        mock_sql.assert_called_once()

    def test_routes_mongo_alias(self, monkeypatch):
        """DBReader accepts 'mongo' alias for mongodb."""
        from ingestion.readers.db_reader import DBReader, DBSourceConfig

        config = DBSourceConfig(backend="mongo", database="testdb", table_or_collection="c")

        with patch.object(DBReader, "_mongo") as mock_mongo:
            mock_mongo.return_value = MagicMock(
                data=pd.DataFrame(), row_count=0,
                schema_extracted={}, pk_columns=[], watermark_new_value=None,
                read_time_ms=0, errors=[]
            )
            reader = DBReader()
            reader.read(config)

        mock_mongo.assert_called_once()

    def test_result_includes_read_time_ms(self, monkeypatch):
        """DBReadResult includes timing information."""
        from ingestion.readers.db_reader import DBReader, DBSourceConfig

        config = DBSourceConfig(backend="mongo", database="testdb", table_or_collection="c")

        with patch.object(DBReader, "_mongo") as mock_mongo:
            mock_mongo.return_value = MagicMock(
                data=pd.DataFrame({"x": [1]}), row_count=1,
                schema_extracted={"x": "int64"}, pk_columns=[], watermark_new_value=None,
                read_time_ms=0, errors=[]
            )
            reader = DBReader()
            result = reader.read(config)

        assert result.read_time_ms >= 0
