"""
scripts/start_kafka_pipeline.py
---------------------------------
Production-ready DIPEX Kafka pipeline launcher.

Usage:
    python scripts/start_kafka_pipeline.py

Environment variables (set in .env or shell):
    KAFKA_BOOTSTRAP_SERVERS  — Kafka broker(s), default: localhost:9092
    KAFKA_GROUP_ID           — Consumer group ID, default: dipex-pipeline
    KAFKA_TOPICS             — Comma-separated topics, default: raw_events
    KAFKA_BATCH_SIZE         — Records per batch, default: 1000
    LLM_PROVIDER             — huggingface | local | fallback
    HF_API_KEY               — HuggingFace API token (if LLM_PROVIDER=huggingface)

Press Ctrl-C or send SIGTERM to shut down gracefully.
"""

from __future__ import annotations

import logging
import os
import sys

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("dipex.startup")


def _load_dotenv() -> None:
    """Load .env file from project root if it exists (no external library needed)."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    env_path = os.path.normpath(env_path)
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:   # don't override existing env
                os.environ[key] = value
    logger.info("Loaded .env from %s", env_path)


def _build_config() -> dict:
    """Build pipeline config from environment variables."""
    topics_raw = os.environ.get("KAFKA_TOPICS", "raw_events")
    topics     = [t.strip() for t in topics_raw.split(",") if t.strip()]

    return {
        "kafka": {
            "bootstrap_servers": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "group_id":          os.environ.get("KAFKA_GROUP_ID",          "dipex-pipeline"),
            "topics":            topics,
            "batch_size":        int(os.environ.get("KAFKA_BATCH_SIZE",    "1000")),
            "auto_offset_reset": os.environ.get("KAFKA_AUTO_OFFSET_RESET", "latest"),
            "value_deserializer": "json",
            # SASL / TLS (populated from env if present)
            "security_protocol": os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
            "sasl_mechanism":    os.environ.get("KAFKA_SASL_MECHANISM",    ""),
            "sasl_username":     os.environ.get("KAFKA_SASL_USERNAME",     ""),
            "sasl_password":     os.environ.get("KAFKA_SASL_PASSWORD",     ""),
        },
        "pipeline": {
            "target_col":   os.environ.get("DIPEX_TARGET_COL"),    # None = unsupervised
            "skip_stages":  [],
            "domain":       os.environ.get("DIPEX_DOMAIN", "default"),
        },
        "llm": {
            "provider":     os.environ.get("LLM_PROVIDER",   "fallback"),
            "hf_api_key":   os.environ.get("HF_API_KEY",     ""),
            "hf_model_name":os.environ.get("HF_MODEL_NAME",  "mistralai/Mistral-7B-Instruct-v0.2"),
        },
        "lag_log_every_n_batches": 50,
        "max_consecutive_errors":  10,
    }


def main() -> int:
    # 1. Load .env (silently ignored if not present)
    _load_dotenv()

    # 2. Build config
    config = _build_config()

    logger.info("=" * 60)
    logger.info("DIPEX Kafka Pipeline — Starting")
    logger.info("  Brokers : %s", config["kafka"]["bootstrap_servers"])
    logger.info("  Topics  : %s", config["kafka"]["topics"])
    logger.info("  Group   : %s", config["kafka"]["group_id"])
    logger.info("  Batch   : %d records", config["kafka"]["batch_size"])
    logger.info("  LLM     : %s", config["llm"]["provider"])
    logger.info("=" * 60)

    # 3. Verify kafka broker is reachable before starting threads
    try:
        from ingestion.connectors.kafka_connector import KafkaConnector
        test_conn = KafkaConnector(config["kafka"])
        if not test_conn.test_connection():
            logger.error(
                "Kafka broker at '%s' is not reachable. "
                "Check KAFKA_BOOTSTRAP_SERVERS.",
                config["kafka"]["bootstrap_servers"],
            )
            return 1
        logger.info("Kafka connectivity: OK")
    except Exception as exc:
        logger.error("Kafka pre-flight check failed: %s", exc)
        logger.info(
            "Tip: Is Kafka running at %s? "
            "Start with: docker run -d -p 9092:9092 apache/kafka:latest",
            config["kafka"]["bootstrap_servers"],
        )
        return 1

    # 4. Start the pipeline (blocks until Ctrl-C / SIGTERM)
    from ingestion.kafka_pipeline import KafkaPipelineRunner
    runner = KafkaPipelineRunner(config)
    runner.start(block=True)   # graceful shutdown is handled internally

    logger.info("DIPEX Kafka Pipeline — Stopped cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
