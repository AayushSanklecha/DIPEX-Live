"""
ingestion/readers/db_reader.py
--------------------------------
Universal database reader supporting 15+ backends.
See implementation_plan.md for full design spec.
"""
from __future__ import annotations
import logging, os, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from ingestion.error_handler import DBConnectionError, DataFormatError

logger = logging.getLogger("dipex.ingestion.readers.db")

def _env(var: str) -> str:
    return os.environ.get(var, "")

@dataclass
class DBSourceConfig:
    backend: str
    database: str = ""
    host: str = "localhost"
    port: Optional[int] = None
    username_env: str = ""
    password_env: str = ""
    dsn_env: str = ""
    table_or_collection: str = ""
    query: str = ""
    watermark_column: str = ""
    watermark_last_value: Any = None
    chunk_size: int = 50_000
    fetch_chunk_size: int = 50_000     # server-side cursor batch size (for 50GB DB reads)
    schema: str = ""
    extra_connect_args: Dict[str, Any] = field(default_factory=dict)
    snowflake_warehouse: str = ""
    snowflake_role: str = ""
    bigquery_project: str = ""
    bigquery_dataset: str = ""
    bigquery_credentials_env: str = ""
    cassandra_keyspace: str = ""
    cassandra_contact_points: List[str] = field(default_factory=lambda: ["localhost"])
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_cypher: str = ""
    max_total_gb: float = 50.0         # hard cap for chunked DB reads

@dataclass
class DBReadResult:
    data: pd.DataFrame
    row_count: int
    schema_extracted: Dict[str, str]
    pk_columns: List[str]
    watermark_new_value: Any
    read_time_ms: float
    errors: List = field(default_factory=list)


class DBReader:
    """Universal database reader. Dispatches to the correct backend adapter."""

    def read(self, config: DBSourceConfig) -> DBReadResult:
        t0 = time.perf_counter()
        backend = config.backend.lower().replace("-", "").replace("_", "")

        # ── [RL] Adaptive chunk-size selection ─────────────────────────────
        _rl_agent = None
        try:
            from ingestion.adaptive_rate_limiter import get_rl_agent as _get_rl
            _rl_agent = _get_rl()
            config.chunk_size = _rl_agent.get_db_chunk_size(backend, config.host)
            logger.debug("[RL] DB chunk_size=%d for %s@%s", config.chunk_size, backend, config.host)
        except Exception:  # noqa: BLE001
            pass

        dispatch = {
            "postgres": self._sql, "postgresql": self._sql,
            "mysql": self._sql, "mariadb": self._sql, "sqlite": self._sql,
            "mssql": self._sql, "sqlserver": self._sql, "oracle": self._sql,
            "snowflake": self._sql, "redshift": self._sql, "clickhouse": self._sql,
            "bigquery": self._bigquery,
            "mongodb": self._mongo, "mongo": self._mongo,
            "redis": self._redis,
            "dynamodb": self._dynamo,
            "cassandra": self._cassandra,
            "neo4j": self._neo4j,
            "couchdb": self._couch,
        }
        fn = dispatch.get(backend)
        if fn is None:
            raise DBConnectionError(f"Unsupported backend: {config.backend!r}")
        try:
            result = fn(config)
            result.read_time_ms = round((time.perf_counter() - t0) * 1000, 2)
            if _rl_agent:
                _rl_agent.record_db_outcome(backend, config.host, config.chunk_size,
                                            success=True, latency_ms=result.read_time_ms)
            return result
        except Exception as exc:
            if _rl_agent:
                elapsed = round((time.perf_counter() - t0) * 1000, 2)
                _rl_agent.record_db_outcome(backend, config.host, config.chunk_size,
                                            success=False, latency_ms=elapsed)
            raise

    def _build_dsn(self, config: DBSourceConfig) -> str:
        if config.dsn_env:
            return _env(config.dsn_env)
        u, p, h, po, db = (
            _env(config.username_env), _env(config.password_env),
            config.host, config.port, config.database,
        )
        b = config.backend.lower()
        if b in ("postgres", "postgresql"):
            return f"postgresql+psycopg2://{u}:{p}@{h}:{po or 5432}/{db}"
        if b in ("mysql", "mariadb"):
            return f"mysql+pymysql://{u}:{p}@{h}:{po or 3306}/{db}"
        if b == "sqlite":
            return f"sqlite:///{db}"
        if b in ("mssql", "sqlserver"):
            return f"mssql+pyodbc://{u}:{p}@{h}:{po or 1433}/{db}?driver=ODBC+Driver+17+for+SQL+Server"
        if b == "oracle":
            return f"oracle+cx_oracle://{u}:{p}@{h}:{po or 1521}/{db}"
        if b == "snowflake":
            import urllib.parse
            acc = config.extra_connect_args.get("account", "")
            dsn = f"snowflake://{u}:{urllib.parse.quote(p)}@{acc}/{db}"
            if config.snowflake_warehouse:
                dsn += f"?warehouse={config.snowflake_warehouse}"
            return dsn
        if b == "redshift":
            return f"redshift+psycopg2://{u}:{p}@{h}:{po or 5439}/{db}"
        if b == "clickhouse":
            return f"clickhouse+native://{u}:{p}@{h}:{po or 9000}/{db}"
        return f"sqlite:///{db}"

    def _sql(self, config: DBSourceConfig) -> DBReadResult:
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            raise DBConnectionError("sqlalchemy not installed")

        dsn = self._build_dsn(config)
        last_exc: Optional[Exception] = None

        # ── Connection retry with exponential backoff ─────────────────────
        for attempt in range(3):
            try:
                engine = create_engine(
                    dsn, pool_size=5, max_overflow=10,
                    connect_args=config.extra_connect_args,
                    execution_options={"stream_results": True},
                )
                # Test the connection before reading
                with engine.connect() as _test_conn:
                    pass
                break  # Connection succeeded
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                import random, time as _time
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "[DBReader] Connection attempt %d failed (%s) — retrying in %.1fs",
                    attempt + 1, exc, wait,
                )
                _time.sleep(wait)
        else:
            raise DBConnectionError(
                f"DB connection failed after 3 attempts: {last_exc}"
            ) from last_exc

        df = pd.DataFrame()
        total_rows = 0
        try:
            with engine.connect() as conn:
                if config.query:
                    sql = config.query
                else:
                    tbl = (f'"{config.schema}"."{config.table_or_collection}"'
                           if config.schema else f'"{config.table_or_collection}"')
                    sql = f"SELECT * FROM {tbl}"
                    if config.watermark_column and config.watermark_last_value is not None:
                        sql += f" WHERE \"{config.watermark_column}\" > '{config.watermark_last_value}'"

                # Use ChunkedParquetWriter for large streaming reads
                try:
                    import uuid as _uuid
                    from ingestion.chunked_writer import ChunkedParquetWriter
                    run_id = _uuid.uuid4().hex[:12]
                    writer = ChunkedParquetWriter(
                        base_dir="data/tmp",
                        dataset_id=(config.table_or_collection or "db_query")[:32],
                        run_id=run_id,
                        max_total_gb=config.max_total_gb,
                    )
                    chunked_iter = pd.read_sql(sql, conn, chunksize=config.fetch_chunk_size)
                    for chunk in chunked_iter:
                        writer.write_chunk(chunk)
                    df = writer.merge_to_single(sample_rows=500_000)
                    total_rows = writer.total_rows
                    writer.cleanup()
                except Exception as chi_exc:  # noqa: BLE001
                    logger.warning("DB ChunkedWriter path failed (%s) — fallback to pd.concat", chi_exc)
                    try:
                        chunks = [c for c in pd.read_sql(sql, conn, chunksize=config.chunk_size)]
                        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                        total_rows = len(df)
                    except Exception as sql_exc:  # noqa: BLE001
                        # Custom query totally failed — fall back to sampled SELECT *
                        if config.query:
                            logger.warning(
                                "[DBReader] Custom query failed (%s) — falling back to LIMIT 1000 sample",
                                sql_exc,
                            )
                            tbl = (f'"{config.schema}"."{config.table_or_collection}"'
                                   if config.schema else f'"{config.table_or_collection}"')
                            fallback_sql = f"SELECT * FROM {tbl} LIMIT 1000"
                            try:
                                df = pd.read_sql(fallback_sql, conn)
                                total_rows = len(df)
                            except Exception as fb_exc:  # noqa: BLE001
                                logger.error("[DBReader] Fallback query also failed: %s", fb_exc)
                        else:
                            raise

        except Exception as exc:
            if any(w in str(exc).lower() for w in ("connect", "refused", "authentication")):
                raise DBConnectionError(f"DB connection failed: {exc}") from exc
            raise DataFormatError(f"SQL error: {exc}") from exc
        finally:
            try:
                engine.dispose()
            except Exception:  # noqa: BLE001
                pass

        # ── Empty result guard ───────────────────────────────────
        if df is None:
            df = pd.DataFrame()
        if df.empty:
            logger.info("[DBReader] Query returned 0 rows — returning empty result (valid, non-fatal)")

        # ── PostgreSQL ARRAY / JSON column expansion ──────────────────────
        if not df.empty:
            df = self._expand_complex_cols(df)

        wm = df[config.watermark_column].max() if config.watermark_column in (df.columns if not df.empty else []) else None
        logger.info("[DBReader] SQL read complete — %d rows (total_fetched=%d)", len(df), total_rows)
        return DBReadResult(df, len(df), {c: str(df[c].dtype) for c in df.columns}, [], wm, 0)

    @staticmethod
    def _expand_complex_cols(df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand columns that hold PostgreSQL ARRAY values (lists) or
        JSON strings that were returned as Python dicts/strings.
        ARRAY columns → join as comma-separated string.
        JSON/dict columns → pd.json_normalize and merge back.
        """
        import json as _json
        for col in list(df.columns):
            try:
                sample = df[col].dropna().head(20)
                if len(sample) == 0:
                    continue
                # Detect list (PostgreSQL ARRAY) columns
                if all(isinstance(v, list) for v in sample):
                    df[col] = df[col].apply(
                        lambda v: ", ".join(str(x) for x in v) if isinstance(v, list) else v
                    )
                    logger.info("[DBReader] ARRAY column '%s' — joined as string", col)
                    continue
                # Detect dict (pg json/jsonb) columns
                if all(isinstance(v, dict) for v in sample):
                    expanded = pd.json_normalize(df[col].apply(
                        lambda v: v if isinstance(v, dict) else {}
                    ).tolist())
                    expanded.columns = [f"{col}.{c}" for c in expanded.columns]
                    expanded.index = df.index
                    df = pd.concat([df.drop(columns=[col]), expanded], axis=1)
                    logger.info("[DBReader] JSON/dict column '%s' — expanded to %d sub-cols", col, len(expanded.columns))
                    continue
                # Detect JSON-string columns
                str_sample = [v for v in sample if isinstance(v, str)]
                if str_sample:
                    parsed = []
                    for v in str_sample[:10]:
                        try:
                            obj = _json.loads(v)
                            if isinstance(obj, (dict, list)):
                                parsed.append(obj)
                        except Exception:  # noqa: BLE001
                            pass
                    if len(parsed) / len(str_sample) >= 0.7:
                        def _safe_json(v):
                            try:
                                return _json.loads(v) if isinstance(v, str) else v
                            except Exception:  # noqa: BLE001
                                return {}
                        objs = df[col].apply(_safe_json)
                        expanded = pd.json_normalize(objs.tolist())
                        expanded.columns = [f"{col}.{c}" for c in expanded.columns]
                        expanded.index = df.index
                        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)
                        logger.info("[DBReader] JSON-string column '%s' expanded", col)
            except Exception:  # noqa: BLE001
                pass
        return df

    def _bigquery(self, config: DBSourceConfig) -> DBReadResult:
        try:
            from google.cloud import bigquery
            cred = _env(config.bigquery_credentials_env)
            if cred: os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred
            client = bigquery.Client(project=config.bigquery_project)
            sql = config.query or f"SELECT * FROM `{config.bigquery_project}.{config.bigquery_dataset}.{config.table_or_collection}`"
            df = client.query(sql).to_dataframe()
            return DBReadResult(df, len(df), {c: str(df[c].dtype) for c in df.columns}, [], None, 0)
        except ImportError:
            raise DBConnectionError("google-cloud-bigquery not installed")
        except Exception as exc:
            raise DBConnectionError(f"BigQuery error: {exc}") from exc

    def _mongo(self, config: DBSourceConfig) -> DBReadResult:
        try:
            from pymongo import MongoClient
            u, p = _env(config.username_env), _env(config.password_env)
            uri = _env(config.dsn_env) or f"mongodb://{u}:{p}@{config.host}:{config.port or 27017}/"
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            filt = {}
            if config.watermark_column and config.watermark_last_value is not None:
                filt[config.watermark_column] = {"$gt": config.watermark_last_value}
            # batch_size() prevents buffering the full collection in RAM
            cursor = (
                client[config.database][config.table_or_collection]
                .find(filt, {"_id": 0})
                .batch_size(config.fetch_chunk_size)
            )
            # Stream into ChunkedParquetWriter for large collections
            try:
                import uuid as _uuid
                from ingestion.chunked_writer import ChunkedParquetWriter
                run_id = _uuid.uuid4().hex[:12]
                writer = ChunkedParquetWriter(
                    base_dir="data/tmp",
                    dataset_id=(config.table_or_collection or "mongo")[:32],
                    run_id=run_id,
                    max_total_gb=config.max_total_gb,
                )
                batch: list = []
                for doc in cursor:
                    batch.append(doc)
                    if len(batch) >= config.fetch_chunk_size:
                        writer.write_chunk(pd.json_normalize(batch))
                        batch.clear()
                if batch:
                    writer.write_chunk(pd.json_normalize(batch))
                df = writer.merge_to_single(sample_rows=500_000)
                writer.cleanup()
            except Exception as mchi_exc:  # noqa: BLE001
                logger.warning("Mongo ChunkedWriter failed (%s) — load all at once", mchi_exc)
                records = list(cursor)
                df = pd.json_normalize(records) if records else pd.DataFrame()
            client.close()
            return DBReadResult(df, len(df), {c: str(df[c].dtype) for c in df.columns}, [], None, 0)
        except ImportError:
            raise DBConnectionError("pymongo not installed")
        except Exception as exc:
            raise DBConnectionError(f"MongoDB error: {exc}") from exc

    def _redis(self, config: DBSourceConfig) -> DBReadResult:
        try:
            import redis as redis_lib
            r = redis_lib.Redis(host=config.host, port=config.port or 6379,
                                password=_env(config.password_env) or None,
                                decode_responses=True, socket_connect_timeout=5)
            pattern = config.query or config.table_or_collection or "*"
            keys = r.keys(pattern)
            records = []
            for k in keys[:100_000]:
                t = r.type(k)
                if t == "hash": records.append({"_key": k, **r.hgetall(k)})
                elif t == "string": records.append({"_key": k, "value": r.get(k)})
                elif t == "list": records.append({"_key": k, "values": r.lrange(k, 0, -1)})
            df = pd.DataFrame(records) if records else pd.DataFrame()
            return DBReadResult(df, len(df), {c: str(df[c].dtype) for c in df.columns}, ["_key"], None, 0)
        except ImportError:
            raise DBConnectionError("redis not installed")
        except Exception as exc:
            raise DBConnectionError(f"Redis error: {exc}") from exc

    def _dynamo(self, config: DBSourceConfig) -> DBReadResult:
        try:
            import boto3
            region = config.extra_connect_args.get("region_name", "us-east-1")
            table = boto3.resource("dynamodb", region_name=region).Table(config.table_or_collection)
            items, kw = [], {}
            while True:
                resp = table.scan(**kw)
                items.extend(resp.get("Items", []))
                last = resp.get("LastEvaluatedKey")
                if not last: break
                kw["ExclusiveStartKey"] = last
            df = pd.json_normalize(items) if items else pd.DataFrame()
            return DBReadResult(df, len(df), {c: str(df[c].dtype) for c in df.columns}, [], None, 0)
        except ImportError:
            raise DBConnectionError("boto3 not installed")
        except Exception as exc:
            raise DBConnectionError(f"DynamoDB error: {exc}") from exc

    def _cassandra(self, config: DBSourceConfig) -> DBReadResult:
        try:
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
            u, p = _env(config.username_env), _env(config.password_env)
            auth = PlainTextAuthProvider(username=u, password=p) if u else None
            cluster = Cluster(config.cassandra_contact_points, auth_provider=auth, connect_timeout=10)
            session = cluster.connect(config.cassandra_keyspace)
            cql = config.query or f"SELECT * FROM {config.table_or_collection}"
            df = pd.DataFrame(list(session.execute(cql)))
            cluster.shutdown()
            return DBReadResult(df, len(df), {c: str(df[c].dtype) for c in df.columns}, [], None, 0)
        except ImportError:
            raise DBConnectionError("cassandra-driver not installed")
        except Exception as exc:
            raise DBConnectionError(f"Cassandra error: {exc}") from exc

    def _neo4j(self, config: DBSourceConfig) -> DBReadResult:
        try:
            from neo4j import GraphDatabase
            u = _env(config.username_env) or "neo4j"
            p = _env(config.password_env) or "neo4j"
            driver = GraphDatabase.driver(config.neo4j_uri, auth=(u, p))
            cypher = config.neo4j_cypher or config.query or f"MATCH (n:{config.table_or_collection}) RETURN n LIMIT 100000"
            records = []
            with driver.session(database=config.database or "neo4j") as sess:
                for r in sess.run(cypher):
                    records.append(dict(r))
            driver.close()
            df = pd.json_normalize(records) if records else pd.DataFrame()
            return DBReadResult(df, len(df), {c: str(df[c].dtype) for c in df.columns}, [], None, 0)
        except ImportError:
            raise DBConnectionError("neo4j not installed")
        except Exception as exc:
            raise DBConnectionError(f"Neo4j error: {exc}") from exc

    def _couch(self, config: DBSourceConfig) -> DBReadResult:
        try:
            import couchdb
            u, p = _env(config.username_env), _env(config.password_env)
            server = couchdb.Server(f"http://{u}:{p}@{config.host}:{config.port or 5984}/")
            db = server[config.database]
            docs = [db[did] for did in db]
            df = pd.json_normalize(docs) if docs else pd.DataFrame()
            return DBReadResult(df, len(df), {c: str(df[c].dtype) for c in df.columns}, ["_id"], None, 0)
        except ImportError:
            raise DBConnectionError("couchdb not installed")
        except Exception as exc:
            raise DBConnectionError(f"CouchDB error: {exc}") from exc

    def list_tables(self, config: DBSourceConfig) -> List[str]:
        b = config.backend.lower().replace("-", "").replace("_", "")
        
        if b in ("postgres", "postgresql", "mysql", "mariadb", "sqlite", "mssql", "sqlserver", "oracle", "snowflake", "redshift", "clickhouse"):
            try:
                from sqlalchemy import inspect, create_engine
                dsn = self._build_dsn(config)
                engine = create_engine(dsn, connect_args={"connect_timeout": 5})
                insp = inspect(engine)
                tables = insp.get_table_names(schema=config.schema or None)
                views = insp.get_view_names(schema=config.schema or None)
                engine.dispose()
                return sorted(list(set(tables + views)))
            except Exception as exc:
                logger.error("Failed to list SQL tables: %s", exc)
                raise
                
        if b in ("mongodb", "mongo"):
            try:
                from pymongo import MongoClient
                u = _env(config.username_env)
                p = _env(config.password_env)
                if u and p:
                    uri = f"mongodb://{u}:{p}@{config.host}:{config.port or 27017}/"
                elif u:
                    uri = f"mongodb://{u}@{config.host}:{config.port or 27017}/"
                else:
                    uri = f"mongodb://{config.host}:{config.port or 27017}/"
                client = MongoClient(uri, serverSelectionTimeoutMS=5000)
                # Force connection to verify auth works
                client.admin.command('ping')
                db = client[config.database]
                collections = db.list_collection_names()
                client.close()
                return sorted(collections)
            except Exception as exc:
                logger.error("Failed to list MongoDB collections: %s", exc)
                raise
                
        if b == "redis":
            try:
                import redis as redis_lib
                r = redis_lib.Redis(host=config.host, port=config.port or 6379,
                                    password=_env(config.password_env) or None,
                                    decode_responses=True, socket_connect_timeout=5)
                # Just return a few top-level patterns or types to prevent overwhelming the UI
                # Usually users filter by key prefix anyway, so we just return a placeholder or root keys 
                keys = r.keys("*")
                prefixes = {k.split(':')[0] for k in keys[:1000] if ':' in k}
                return sorted(list(prefixes)) if prefixes else ["*"]
            except Exception as exc:
                logger.error("Failed to list Redis keys/prefixes: %s", exc)
                return []
                
        if b == "neo4j":
            try:
                from neo4j import GraphDatabase
                u = _env(config.username_env) or "neo4j"
                p = _env(config.password_env) or "neo4j"
                driver = GraphDatabase.driver(config.neo4j_uri, auth=(u, p))
                labels = []
                with driver.session(database=config.database or "neo4j") as sess:
                    res = sess.run("CALL db.labels()")
                    labels = [record[0] for record in res]
                driver.close()
                return sorted(labels)
            except Exception as exc:
                logger.error("Failed to list Neo4j labels: %s", exc)
                return []
                
        if b == "dynamodb":
            try:
                import boto3
                region = config.extra_connect_args.get("region_name", "us-east-1")
                client = boto3.client("dynamodb", region_name=region)
                return client.list_tables().get("TableNames", [])
            except Exception as exc:
                logger.error("Failed to list DynamoDB tables: %s", exc)
                return []

        # Default fallback for unimplemented list_tables backends
        logger.warning(f"list_tables not explicitly implemented for {b}, returning empty list.")
        return []

    def extract_schema(self, config: DBSourceConfig) -> Dict[str, Any]:
        b = config.backend.lower()
        if b in ("postgres", "postgresql", "mysql", "sqlite", "mssql", "oracle"):
            try:
                from sqlalchemy import inspect
                engine_obj = __import__("sqlalchemy").create_engine(self._build_dsn(config))
                insp = inspect(engine_obj)
                cols = insp.get_columns(config.table_or_collection, schema=config.schema or None)
                pks  = insp.get_pk_constraint(config.table_or_collection, schema=config.schema or None)
                engine_obj.dispose()
                return {"columns": [{"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True)} for c in cols],
                        "primary_keys": pks.get("constrained_columns", [])}
            except Exception as exc:
                logger.warning("Schema extraction failed: %s", exc)
        return {}
