"""
ingestion/readers/stream_reader.py
------------------------------------
Universal streaming data reader.

Supported sources
-----------------
Apache Kafka / Redpanda (confluent-kafka)
WebSocket streams (websockets)
IoT / Event Hub stubs (generic callback-based)

Design contracts
----------------
- Event-time processing with configurable watermark (seconds)
- Sliding & tumbling window implementations
- Late data buffering and discard policy
- Backpressure detection (consumer lag monitoring)
- Consumer lag threshold alert
- Each closed window → returns an immutable DataFrame (snapshot)
- Graceful shutdown on signal / timeout
- All errors → StreamLagError / DataFormatError — never crash
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import pandas as pd

from ingestion.error_handler import DataFormatError, StreamLagError

logger = logging.getLogger("dipex.ingestion.readers.stream")


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class KafkaSourceConfig:
    brokers: str = "kafka:29092"
    topic: str = ""
    group_id: str = "dipex-consumer"
    auto_offset_reset: str = "earliest"
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username_env: str = ""
    sasl_password_env: str = ""
    max_messages: int = 100_000
    poll_timeout_s: float = 1.0
    consumer_lag_warn_threshold: int = 50_000
    consumer_lag_error_threshold: int = 500_000


@dataclass
class WindowConfig:
    strategy: str = "tumbling"       # tumbling | sliding
    window_size_s: float = 60.0      # window duration in seconds
    slide_step_s: float = 30.0       # for sliding windows only
    watermark_delay_s: float = 5.0   # late-arrival tolerance
    event_time_field: Optional[str] = None  # field name for event time


@dataclass
class StreamReadResult:
    data: pd.DataFrame
    row_count: int
    window_start: str
    window_end: str
    late_dropped: int
    consumer_lag: int
    read_time_ms: float
    errors: List = field(default_factory=list)


# ── Tumbling Window ───────────────────────────────────────────────────────────

class TumblingWindow:
    """Collects events into fixed-size, non-overlapping time windows."""

    def __init__(self, window_size_s: float, watermark_delay_s: float = 5.0):
        self.size_s    = window_size_s
        self.delay_s   = watermark_delay_s
        self._buffer: List[Dict] = []
        self._window_start = time.time()
        self._late_dropped = 0

    def add(self, record: Dict, event_time: Optional[float] = None) -> None:
        ts = event_time or time.time()
        watermark = time.time() - self.delay_s
        if ts < watermark:
            self._late_dropped += 1
            logger.debug("Late record dropped (age=%.1fs)", time.time() - ts)
            return
        self._buffer.append({**record, "_event_time": ts})

    def should_close(self) -> bool:
        return (time.time() - self._window_start) >= self.size_s

    def close(self) -> Tuple[List[Dict], float, float, int]:
        """Close window, return (records, start, end, late_dropped)."""
        start = self._window_start
        end = time.time()
        records = list(self._buffer)
        late = self._late_dropped
        self._buffer.clear()
        self._window_start = end
        self._late_dropped = 0
        return records, start, end, late


class SlidingWindow:
    """Overlapping windows; each step produces a new snapshot."""

    def __init__(self, window_size_s: float, slide_step_s: float, watermark_delay_s: float = 5.0):
        self.size_s   = window_size_s
        self.step_s   = slide_step_s
        self.delay_s  = watermark_delay_s
        self._events: List[Tuple[float, Dict]] = []
        self._last_emit = time.time()
        self._late_dropped = 0

    def add(self, record: Dict, event_time: Optional[float] = None) -> None:
        ts = event_time or time.time()
        if ts < time.time() - self.delay_s:
            self._late_dropped += 1
            return
        self._events.append((ts, record))

    def should_emit(self) -> bool:
        return (time.time() - self._last_emit) >= self.step_s

    def emit(self) -> Tuple[List[Dict], float, float, int]:
        now = time.time()
        window_start = now - self.size_s
        records = [r for ts, r in self._events if ts >= window_start]
        # Expire old events
        self._events = [(ts, r) for ts, r in self._events if ts >= window_start - self.delay_s]
        late = self._late_dropped
        self._late_dropped = 0
        self._last_emit = now
        return records, window_start, now, late


# ── Kafka Reader ──────────────────────────────────────────────────────────────

class StreamReader:
    """
    Universal streaming reader.

    Usage — Kafka tumbling windows::

        cfg = KafkaSourceConfig(brokers="localhost:9092", topic="events")
        window_cfg = WindowConfig(strategy="tumbling", window_size_s=30)
        reader = StreamReader()
        for snapshot in reader.read_kafka(cfg, window_cfg):
            # snapshot: StreamReadResult with .data DataFrame
            process(snapshot)
    """

    def read_kafka(
        self,
        config: KafkaSourceConfig,
        window_cfg: WindowConfig,
        max_windows: int = 1000,
        transform_fn: Optional[Callable[[bytes], Dict]] = None,
    ) -> Generator[StreamReadResult, None, None]:
        """Yield one StreamReadResult per closed window."""
        try:
            from confluent_kafka import Consumer, KafkaError, TopicPartition
        except ImportError:
            raise DataFormatError(
                "confluent-kafka not installed — run: pip install confluent-kafka"
            )
        import os

        conf: Dict[str, Any] = {
            "bootstrap.servers": config.brokers,
            "group.id": config.group_id,
            "auto.offset.reset": config.auto_offset_reset,
            "enable.auto.commit": True,
            "session.timeout.ms": 30_000,
        }
        if config.security_protocol != "PLAINTEXT":
            conf["security.protocol"] = config.security_protocol
        if config.sasl_mechanism:
            conf["sasl.mechanisms"] = config.sasl_mechanism
            conf["sasl.username"]   = os.environ.get(config.sasl_username_env, "")
            conf["sasl.password"]   = os.environ.get(config.sasl_password_env, "")

        consumer = Consumer(conf)
        consumer.subscribe([config.topic])
        logger.info("Kafka consumer subscribed to topic '%s'", config.topic)

        # Select window type
        if window_cfg.strategy == "sliding":
            window: Any = SlidingWindow(
                window_cfg.window_size_s, window_cfg.slide_step_s, window_cfg.watermark_delay_s
            )
        else:
            window = TumblingWindow(window_cfg.window_size_s, window_cfg.watermark_delay_s)

        messages_consumed = 0
        windows_emitted = 0
        t0 = time.perf_counter()

        try:
            while windows_emitted < max_windows and messages_consumed < config.max_messages:
                msg = consumer.poll(config.poll_timeout_s)

                if msg is None:
                    # No message — check if window should close
                    pass
                elif msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug("Partition EOF reached.")
                    else:
                        raise StreamLagError(f"Kafka error: {msg.error()}")
                else:
                    # Parse message value
                    raw = msg.value()
                    try:
                        if transform_fn:
                            record = transform_fn(raw)
                        else:
                            record = json.loads(raw.decode("utf-8", errors="replace"))
                    except Exception:  # noqa: BLE001
                        record = {"_raw": str(raw)}

                    # Extract event time
                    evt_ts: Optional[float] = None
                    if window_cfg.event_time_field and window_cfg.event_time_field in record:
                        try:
                            evt_ts = pd.Timestamp(record[window_cfg.event_time_field]).timestamp()
                        except Exception:  # noqa: BLE001
                            evt_ts = None
                    if evt_ts is None:
                        # Use Kafka message timestamp
                        ts_type, ts_val = msg.timestamp()
                        evt_ts = ts_val / 1000.0 if ts_type != 0 else time.time()

                    window.add(record, event_time=evt_ts)
                    messages_consumed += 1

                # Check consumer lag
                self._check_lag(consumer, config.topic, config)

                # Emit window?
                time_up = (
                    window.should_close() if isinstance(window, TumblingWindow)
                    else window.should_emit()
                )
                capacity_reached = messages_consumed >= config.max_messages
                should_emit = time_up or capacity_reached

                if should_emit:
                    if isinstance(window, TumblingWindow):
                        records, w_start, w_end, late = window.close()
                    else:
                        records, w_start, w_end, late = window.emit()

                    if records:
                        df = pd.json_normalize(records)
                    else:
                        df = pd.DataFrame()

                    elapsed = (time.perf_counter() - t0) * 1000
                    yield StreamReadResult(
                        data=df, row_count=len(df),
                        window_start=datetime.fromtimestamp(w_start, tz=timezone.utc).isoformat(),
                        window_end=datetime.fromtimestamp(w_end, tz=timezone.utc).isoformat(),
                        late_dropped=late, consumer_lag=0,
                        read_time_ms=round(elapsed, 2),
                    )
                    windows_emitted += 1
                    t0 = time.perf_counter()
                    logger.info(
                        "Window %d emitted: %d records, %d late dropped",
                        windows_emitted, len(records), late
                    )
                    if capacity_reached:
                        break  # Stop consuming, we hit max_messages

        except KeyboardInterrupt:
            logger.info("Kafka consumer interrupted by user.")
        finally:
            consumer.close()
            logger.info(
                "Kafka consumer closed. Consumed %d messages, emitted %d windows.",
                messages_consumed, windows_emitted,
            )

    @staticmethod
    def _check_lag(consumer: Any, topic: str, config: KafkaSourceConfig) -> int:
        """Check consumer lag and warn/raise if thresholds exceeded."""
        try:
            watermarks = {
                tp: consumer.get_watermark_offsets(tp, timeout=1.0)
                for tp in consumer.assignment()
            }
            lag = sum(
                max(0, high - consumer.position([tp])[0].offset)
                for tp, (low, high) in watermarks.items()
            )
            if lag >= config.consumer_lag_error_threshold:
                raise StreamLagError(
                    f"Consumer lag {lag:,} exceeds error threshold {config.consumer_lag_error_threshold:,}"
                )
            if lag >= config.consumer_lag_warn_threshold:
                logger.warning("Consumer lag %d exceeds warn threshold", lag)
            return lag
        except (StreamLagError, AttributeError):
            raise
        except Exception:  # noqa: BLE001
            return -1

    # ── WebSocket Reader ──────────────────────────────────────────────────────

    def read_websocket(
        self,
        uri: str,
        window_cfg: WindowConfig,
        max_windows: int = 100,
        transform_fn: Optional[Callable[[str], Dict]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Generator[StreamReadResult, None, None]:
        """Read from a WebSocket stream using windowed snapshots."""
        try:
            import asyncio
            import websockets as ws_lib
        except ImportError:
            raise DataFormatError("websockets not installed — run: pip install websockets")

        record_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()

        async def _ws_consumer():
            async with ws_lib.connect(uri, extra_headers=extra_headers or {}) as ws:
                while not stop_event.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        record: Dict
                        if transform_fn:
                            record = transform_fn(msg)
                        else:
                            try:
                                record = json.loads(msg)
                            except Exception:  # noqa: BLE001
                                record = {"raw": msg}
                        record_queue.put(record)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("WebSocket receive error: %s", exc)
                        break

        def _run_loop():
            asyncio.run(_ws_consumer())

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()
        logger.info("WebSocket consumer started: %s", uri)

        window = TumblingWindow(window_cfg.window_size_s, window_cfg.watermark_delay_s)
        windows_emitted = 0
        t0 = time.perf_counter()

        while windows_emitted < max_windows:
            try:
                while True:
                    try:
                        record = record_queue.get_nowait()
                        window.add(record)
                    except queue.Empty:
                        break
            except Exception:  # noqa: BLE001
                pass

            if window.should_close():
                records, w_start, w_end, late = window.close()
                if records:
                    df = pd.json_normalize(records)
                    yield StreamReadResult(
                        data=df, row_count=len(df),
                        window_start=datetime.fromtimestamp(w_start, tz=timezone.utc).isoformat(),
                        window_end=datetime.fromtimestamp(w_end, tz=timezone.utc).isoformat(),
                        late_dropped=late, consumer_lag=0,
                        read_time_ms=round((time.perf_counter() - t0) * 1000, 2),
                    )
                    windows_emitted += 1
                    t0 = time.perf_counter()
            else:
                time.sleep(0.1)

        stop_event.set()
        thread.join(timeout=5)

    # ── IoT / Generic Event Stream ────────────────────────────────────────────

    def collect_events(
        self,
        events: List[Dict],
        window_cfg: WindowConfig,
    ) -> List[StreamReadResult]:
        """
        Process a list of in-memory events (IoT payloads, event hub batches)
        through windowing and return StreamReadResult snapshots.
        """
        if window_cfg.strategy == "sliding":
            window: Any = SlidingWindow(window_cfg.window_size_s, window_cfg.slide_step_s, window_cfg.watermark_delay_s)
        else:
            window = TumblingWindow(window_cfg.window_size_s, window_cfg.watermark_delay_s)

        results: List[StreamReadResult] = []
        t0 = time.perf_counter()

        for record in events:
            evt_ts = None
            if window_cfg.event_time_field and window_cfg.event_time_field in record:
                try:
                    evt_ts = pd.Timestamp(record[window_cfg.event_time_field]).timestamp()
                except Exception:  # noqa: BLE001
                    pass
            window.add(record, event_time=evt_ts)

        # Flush remaining
        flush_fn = window.close if isinstance(window, TumblingWindow) else window.emit
        records, w_start, w_end, late = flush_fn()
        if records:
            df = pd.json_normalize(records)
            results.append(StreamReadResult(
                data=df, row_count=len(df),
                window_start=datetime.fromtimestamp(w_start, tz=timezone.utc).isoformat(),
                window_end=datetime.fromtimestamp(w_end, tz=timezone.utc).isoformat(),
                late_dropped=late, consumer_lag=0,
                read_time_ms=round((time.perf_counter() - t0) * 1000, 2),
            ))
        return results


# ── Redpanda Support ──────────────────────────────────────────────────────────
# Redpanda is fully Kafka API-compatible. The only difference is the broker address.
# Use KafkaSourceConfig with the Redpanda broker URL.

@dataclass
class RedpandaSourceConfig(KafkaSourceConfig):
    """
    Redpanda is wire-compatible with Kafka.
    Use this config to make intent explicit; internally uses the same confluent-kafka consumer.
    Typical broker format: "localhost:9092" or "redpanda.internal:9092"
    """
    # Redpanda-specific tuning: schema registry endpoint for Avro/Protobuf topics
    schema_registry_url: Optional[str] = None
    use_schema_registry: bool = False


def make_redpanda_reader() -> StreamReader:
    """Factory: returns a StreamReader configured to consume from Redpanda.
    Internally identical to Kafka since Redpanda is Kafka-API-compatible."""
    return StreamReader()


# ── Azure Event Hub Reader ────────────────────────────────────────────────────

@dataclass
class EventHubConfig:
    """Azure Event Hub connection configuration."""
    connection_string_env: str = "AZURE_EH_CONN_STRING"   # env var holding full connection string
    eventhub_name: str = ""
    consumer_group: str = "$Default"
    max_wait_time_s: float = 5.0
    max_events_per_partition: int = 1000


class EventHubReader:
    """
    Azure Event Hub consumer using the azure-eventhub SDK.
    Falls back gracefully if SDK is not installed (stub mode for testing).

    Each batch call collects events from all partitions, applies windowing,
    and returns immutable StreamReadResult snapshots — identical contract
    to the Kafka reader.

    Usage::

        cfg = EventHubConfig(eventhub_name="telemetry")
        window_cfg = WindowConfig(strategy="tumbling", window_size_s=60)
        reader = EventHubReader()
        for snapshot in reader.read(cfg, window_cfg):
            process(snapshot)
    """

    def read(
        self,
        config: EventHubConfig,
        window_cfg: WindowConfig,
        max_windows: int = 100,
        transform_fn: Optional[Callable[[Any], Dict]] = None,
    ) -> Generator[StreamReadResult, None, None]:
        """Consume from Azure Event Hub and yield windowed snapshots."""
        import os
        conn_str = os.environ.get(config.connection_string_env, "")
        if not conn_str:
            raise DataFormatError(
                f"Azure Event Hub connection string not found in env var '{config.connection_string_env}'"
            )

        try:
            from azure.eventhub import EventHubConsumerClient
        except ImportError:
            raise DataFormatError(
                "azure-eventhub SDK not installed — run: pip install azure-eventhub"
            )

        if window_cfg.strategy == "sliding":
            window: Any = SlidingWindow(window_cfg.window_size_s, window_cfg.slide_step_s,
                                         window_cfg.watermark_delay_s)
        else:
            window = TumblingWindow(window_cfg.window_size_s, window_cfg.watermark_delay_s)

        record_queue: queue.Queue = queue.Queue()
        windows_emitted = 0
        stop_event = threading.Event()

        def _on_event(partition_ctx, event):
            try:
                raw = event.body_as_str(encoding="UTF-8")
                if transform_fn:
                    record = transform_fn(event)
                else:
                    try:
                        record = json.loads(raw)
                    except Exception:  # noqa: BLE001
                        record = {"_raw": raw, "partition": partition_ctx.partition_id}
                record["_eh_enqueued_time"] = str(event.enqueued_time)
                record["_eh_sequence_number"] = event.sequence_number
                record_queue.put(record)
            except Exception as exc:  # noqa: BLE001
                logger.warning("EventHub event parse error: %s", exc)

        client = EventHubConsumerClient.from_connection_string(
            conn_str, consumer_group=config.consumer_group,
            eventhub_name=config.eventhub_name,
        )

        recv_thread = threading.Thread(
            target=lambda: client.receive(
                on_event=_on_event,
                max_wait_time=config.max_wait_time_s,
            ),
            daemon=True,
        )
        recv_thread.start()
        logger.info("EventHub consumer started: %s (group=%s)", config.eventhub_name, config.consumer_group)

        t0 = time.perf_counter()
        try:
            while windows_emitted < max_windows and not stop_event.is_set():
                while True:
                    try:
                        record = record_queue.get_nowait()
                        evt_ts: Optional[float] = None
                        if window_cfg.event_time_field and window_cfg.event_time_field in record:
                            try:
                                evt_ts = pd.Timestamp(record[window_cfg.event_time_field]).timestamp()
                            except Exception:  # noqa: BLE001
                                pass
                        window.add(record, event_time=evt_ts)
                    except queue.Empty:
                        break

                should_emit = (window.should_close() if isinstance(window, TumblingWindow)
                               else window.should_emit())
                if should_emit:
                    emit_fn = window.close if isinstance(window, TumblingWindow) else window.emit
                    records, w_start, w_end, late = emit_fn()
                    if records:
                        df = pd.json_normalize(records)
                        yield StreamReadResult(
                            data=df, row_count=len(df),
                            window_start=datetime.fromtimestamp(w_start, tz=timezone.utc).isoformat(),
                            window_end=datetime.fromtimestamp(w_end, tz=timezone.utc).isoformat(),
                            late_dropped=late, consumer_lag=0,
                            read_time_ms=round((time.perf_counter() - t0) * 1000, 2),
                        )
                        windows_emitted += 1
                        t0 = time.perf_counter()
                else:
                    time.sleep(0.05)
        finally:
            stop_event.set()
            client.close()
            logger.info("EventHub consumer closed. %d windows emitted.", windows_emitted)


# ── IoT Stream Reader ─────────────────────────────────────────────────────────

@dataclass
class IoTStreamConfig:
    """
    Configuration for IoT device stream ingestion.
    Supports MQTT, HTTP polling, and direct callback registration.
    """
    device_ids: List[str] = field(default_factory=list)
    protocol: str = "mqtt"              # mqtt | http_poll | callback
    broker_host: str = "localhost"
    broker_port: int = 1883
    topics: List[str] = field(default_factory=list)              # MQTT topic filters
    poll_url_template: str = ""         # e.g. "http://iot-gw/api/device/{device_id}/latest"
    poll_interval_s: float = 5.0
    username_env: str = "IOT_USERNAME"
    password_env: str = "IOT_PASSWORD"
    tls: bool = False
    heartbeat_timeout_s: float = 30.0  # Alert if device silent longer than this


class IoTStreamReader:
    """
    IoT stream consumer with per-device heartbeat monitoring.

    Handles MQTT topics (via paho-mqtt), HTTP polling of IoT gateway endpoints,
    and direct callback registration for custom device drivers.
    All events are routed through windowing to produce immutable DataFrame snapshots.

    Device health tracking
    ----------------------
    - Tracks last seen timestamp per device_id
    - Emits WARN log if device exceeds heartbeat_timeout_s
    - Associates each record with its device_id and ingestion timestamp

    Usage::

        cfg = IoTStreamConfig(topics=["sensors/#"], broker_host="mqtt.factory.local")
        window_cfg = WindowConfig(strategy="tumbling", window_size_s=30, event_time_field="ts")
        reader = IoTStreamReader()
        for snapshot in reader.read_mqtt(cfg, window_cfg):
            process(snapshot)
    """

    def __init__(self) -> None:
        self._last_seen: Dict[str, float] = {}   # device_id → epoch seconds

    def read_mqtt(
        self,
        config: IoTStreamConfig,
        window_cfg: WindowConfig,
        max_windows: int = 100,
        transform_fn: Optional[Callable[[str, bytes], Dict]] = None,
    ) -> Generator[StreamReadResult, None, None]:
        """Consume MQTT topics and yield windowed snapshots."""
        import os
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise DataFormatError("paho-mqtt not installed — run: pip install paho-mqtt")

        record_queue: queue.Queue = queue.Queue()

        def _on_message(client, userdata, msg):
            try:
                payload = msg.payload
                if transform_fn:
                    record = transform_fn(msg.topic, payload)
                else:
                    try:
                        record = json.loads(payload.decode("utf-8", errors="replace"))
                    except Exception:  # noqa: BLE001
                        record = {"_raw": payload.decode("utf-8", errors="replace")}
                record.setdefault("_mqtt_topic", msg.topic)
                record.setdefault("_ingestion_ts", time.time())

                # Update heartbeat
                device_id = record.get("device_id") or msg.topic.split("/")[-1]
                self._last_seen[device_id] = time.time()

                record_queue.put(record)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MQTT message parse error: %s", exc)

        client = mqtt.Client()
        username = os.environ.get(config.username_env, "")
        password = os.environ.get(config.password_env, "")
        if username:
            client.username_pw_set(username, password)
        if config.tls:
            client.tls_set()
        client.on_message = _on_message
        client.connect(config.broker_host, config.broker_port, keepalive=60)
        for topic in config.topics or ["#"]:
            client.subscribe(topic)
        client.loop_start()
        logger.info("MQTT consumer started: %s:%d topics=%s", config.broker_host, config.broker_port, config.topics)

        window = TumblingWindow(window_cfg.window_size_s, window_cfg.watermark_delay_s)
        windows_emitted = 0
        t0 = time.perf_counter()

        try:
            while windows_emitted < max_windows:
                while True:
                    try:
                        record = record_queue.get_nowait()
                        evt_ts: Optional[float] = None
                        if window_cfg.event_time_field and window_cfg.event_time_field in record:
                            try:
                                evt_ts = pd.Timestamp(record[window_cfg.event_time_field]).timestamp()
                            except Exception:  # noqa: BLE001
                                pass
                        window.add(record, event_time=evt_ts)
                    except queue.Empty:
                        break

                # Heartbeat check
                self._check_heartbeats(config.heartbeat_timeout_s)

                if window.should_close():
                    records, w_start, w_end, late = window.close()
                    if records:
                        df = pd.json_normalize(records)
                        yield StreamReadResult(
                            data=df, row_count=len(df),
                            window_start=datetime.fromtimestamp(w_start, tz=timezone.utc).isoformat(),
                            window_end=datetime.fromtimestamp(w_end, tz=timezone.utc).isoformat(),
                            late_dropped=late, consumer_lag=0,
                            read_time_ms=round((time.perf_counter() - t0) * 1000, 2),
                        )
                        windows_emitted += 1
                        t0 = time.perf_counter()
                else:
                    time.sleep(0.05)
        finally:
            client.loop_stop()
            client.disconnect()

    def read_http_poll(
        self,
        config: IoTStreamConfig,
        window_cfg: WindowConfig,
        max_windows: int = 100,
    ) -> Generator[StreamReadResult, None, None]:
        """Poll IoT gateway HTTP endpoints per device and window results."""
        import requests
        window = TumblingWindow(window_cfg.window_size_s, window_cfg.watermark_delay_s)
        windows_emitted = 0
        t0 = time.perf_counter()

        while windows_emitted < max_windows:
            for device_id in config.device_ids:
                url = config.poll_url_template.format(device_id=device_id)
                try:
                    resp = requests.get(url, timeout=5.0)
                    if resp.ok:
                        record = resp.json()
                        record.setdefault("device_id", device_id)
                        record.setdefault("_poll_ts", time.time())
                        self._last_seen[device_id] = time.time()
                        window.add(record)
                    else:
                        logger.warning("IoT poll for device %s returned %d", device_id, resp.status_code)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("IoT poll failed for device %s: %s", device_id, exc)

            self._check_heartbeats(config.heartbeat_timeout_s)

            if window.should_close():
                records, w_start, w_end, late = window.close()
                if records:
                    df = pd.json_normalize(records)
                    yield StreamReadResult(
                        data=df, row_count=len(df),
                        window_start=datetime.fromtimestamp(w_start, tz=timezone.utc).isoformat(),
                        window_end=datetime.fromtimestamp(w_end, tz=timezone.utc).isoformat(),
                        late_dropped=late, consumer_lag=0,
                        read_time_ms=round((time.perf_counter() - t0) * 1000, 2),
                    )
                    windows_emitted += 1
                    t0 = time.perf_counter()

            time.sleep(config.poll_interval_s)

    def _check_heartbeats(self, timeout_s: float) -> None:
        now = time.time()
        for device_id, last in self._last_seen.items():
            if (now - last) > timeout_s:
                logger.warning(
                    "IoT heartbeat timeout: device '%s' silent for %.0fs",
                    device_id, now - last,
                )

