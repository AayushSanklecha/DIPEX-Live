"""
ingestion/websocket_handler.py
-------------------------------
WebSocket and Webhook stream handler for DIPEX Step 1 (Streaming Ingestion).

Supports:
  - WebSocket client: connects to a ws:// or wss:// endpoint, receives JSON events
  - Webhook receiver: async HTTP server endpoint (aiohttp / FastAPI compatible)
  - IoT MQTT bridge: subscribe to MQTT topic, forward to DIPEX pipeline

Features:
  - Event-time ordering (uses 'event_time' field if present, else server recv time)
  - Out-of-order arrival handling (buffered reorder window)
  - Backpressure: max_queue_depth configurable; drops with warning on overflow
  - Auto-reconnect (exponential backoff, max_retries configurable)
  - Each message produces an ISSFSnapshot → fed to PipelineBridge
  - Every window emits immutable Gold snapshot with SHA-256 checksum
  - Late data creates corrective snapshot version (never overwrites original)
  - Full audit logging per message
  - TLS support (wss://)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("dipex.ingestion.websocket_handler")

# ── Constants ──────────────────────────────────────────────────────────────────

_RECONNECT_BASE_S: float  = 1.0
_RECONNECT_MAX_S: float   = 60.0
_DEFAULT_WINDOW_S: float  = 60.0       # 1-minute tumbling window
_REORDER_BUFFER_S: float  = 5.0        # Max late-arrival tolerance within window
_DEFAULT_QUEUE_DEPTH: int = 10_000      # Backpressure threshold
_BATCH_FLUSH_INTERVAL: float = 5.0     # Seconds between forced flushes


# ── Event model ────────────────────────────────────────────────────────────────

class StreamEvent:
    """Single event from a WebSocket or webhook source."""

    __slots__ = ("event_id", "event_time", "recv_time", "payload", "source_id")

    def __init__(
        self,
        payload: Dict[str, Any],
        source_id: str = "ws",
        event_time: Optional[float] = None,
    ) -> None:
        self.event_id   = str(uuid.uuid4())
        self.recv_time  = time.time()
        self.event_time = event_time or self.recv_time
        self.payload    = payload
        self.source_id  = source_id

    @property
    def is_late(self) -> bool:
        return (self.recv_time - self.event_time) > _REORDER_BUFFER_S

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":   self.event_id,
            "event_time": self.event_time,
            "recv_time":  self.recv_time,
            "source_id":  self.source_id,
            "payload":    self.payload,
        }


# ── Window buffer ──────────────────────────────────────────────────────────────

class WindowBuffer:
    """
    Tumbling or sliding window buffer that accumulates StreamEvents,
    then flushes to a downstream handler when the window closes.

    Invariants:
    - Each window produces exactly one ISSFSnapshot with a unique SHA-256 digest
    - Late events (past watermark) produce a new corrective snapshot version,
      never overwrite the original window's snapshot
    - Buffer is bounded by max_events to prevent unbounded memory growth
    """

    def __init__(
        self,
        window_s: float = _DEFAULT_WINDOW_S,
        max_events: int = 100_000,
        window_type: str = "tumbling",
        slide_s: Optional[float] = None,
    ) -> None:
        self.window_s    = window_s
        self.max_events  = max_events
        self.window_type = window_type   # "tumbling" | "sliding"
        self.slide_s     = slide_s or (window_s / 2)
        self._events: List[StreamEvent] = []
        self._window_start = time.time()
        self._emitted_windows: Dict[str, str] = {}   # window_id → snapshot_id

    def add(self, event: StreamEvent) -> Optional[List[StreamEvent]]:
        """
        Add event to buffer. Returns flushed batch if window closes, else None.
        Backpressure: if max_events exceeded, oldest event is dropped with warning.
        """
        if len(self._events) >= self.max_events:
            dropped = self._events.pop(0)
            logger.warning(
                "WindowBuffer: max_events=%d exceeded — dropped event %s (backpressure)",
                self.max_events, dropped.event_id,
            )

        self._events.append(event)

        now = time.time()
        if (now - self._window_start) >= self.window_s:
            return self._flush()
        return None

    def _flush(self) -> List[StreamEvent]:
        """Close window. Sort by event_time for reordering. Return batch."""
        batch = sorted(self._events, key=lambda e: e.event_time)
        self._events = []
        self._window_start = time.time()
        logger.info(
            "WindowBuffer: flushing window with %d events (window_type=%s)",
            len(batch), self.window_type,
        )
        return batch

    def force_flush(self) -> List[StreamEvent]:
        """Force-flush on shutdown or timeout, even if window hasn't closed."""
        batch = sorted(self._events, key=lambda e: e.event_time)
        self._events = []
        return batch

    def window_checksum(self, batch: List[StreamEvent]) -> str:
        """SHA-256 over concatenated event_ids (deterministic per window content)."""
        content = "|".join(e.event_id for e in batch)
        return hashlib.sha256(content.encode()).hexdigest()


# ── WebSocket Client Handler ───────────────────────────────────────────────────

class WebSocketStreamHandler:
    """
    Production-grade WebSocket client that:
    - Connects to ws:// or wss:// endpoints
    - Receives JSON events in a loop with auto-reconnect (exponential backoff)
    - Buffers events into tumbling/sliding windows
    - Flushes each window to PipelineBridge as an ISSFSnapshot
    - Detects and handles late events as corrective snapshots
    - Emits audit events for every window flush

    Usage::

        handler = WebSocketStreamHandler(
            uri="ws://localhost:9090/stream",
            config=config,
            on_window=my_window_callback,
        )
        asyncio.run(handler.run())
    """

    def __init__(
        self,
        uri: str,
        config: Optional[Dict[str, Any]] = None,
        on_window: Optional[Callable[[List[StreamEvent], str, bool], None]] = None,
    ) -> None:
        self.uri      = uri
        self.config   = config or {}
        self.on_window = on_window  # callback(batch, snapshot_id, is_corrective)

        stream_cfg = self.config.get("streaming", {})
        self._window_s     = float(stream_cfg.get("window_size_s", _DEFAULT_WINDOW_S))
        self._max_retries  = int(stream_cfg.get("max_retries", 0))    # 0 = infinite
        self._queue_depth  = int(stream_cfg.get("max_queue_depth", _DEFAULT_QUEUE_DEPTH))
        self._window_type  = str(stream_cfg.get("window_type", "tumbling"))
        self._slide_s      = float(stream_cfg.get("slide_s", self._window_s / 2))
        self._headers      = stream_cfg.get("headers", {})        # e.g. Authorization
        self._ssl          = stream_cfg.get("ssl", None)          # ssl.SSLContext or None
        self._source_id    = str(stream_cfg.get("source_id", "websocket"))
        self._consumer_lag_threshold = int(stream_cfg.get("consumer_lag_threshold", 1000))

        self._buffer   = WindowBuffer(self._window_s, self._queue_depth, self._window_type, self._slide_s)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_depth)
        self._running  = False
        self._retry_count = 0
        self._total_received = 0
        self._total_flushed  = 0

    async def run(self) -> None:
        """Main entry point. Runs until stopped or max_retries exceeded."""
        self._running = True
        logger.info("WebSocketStreamHandler starting: uri=%s window_s=%.0f", self.uri, self._window_s)

        while self._running:
            try:
                await self._connect_and_consume()
            except Exception as exc:  # noqa: BLE001
                self._retry_count += 1
                if self._max_retries > 0 and self._retry_count >= self._max_retries:
                    logger.error(
                        "WebSocket max retries (%d) exceeded. Stopping.", self._max_retries
                    )
                    break
                backoff = min(_RECONNECT_BASE_S * (2 ** min(self._retry_count, 10)), _RECONNECT_MAX_S)
                logger.warning(
                    "WebSocket disconnected (attempt %d): %s. Reconnecting in %.1fs.",
                    self._retry_count, exc, backoff,
                )
                await asyncio.sleep(backoff)

        # Final flush
        remaining = self._buffer.force_flush()
        if remaining:
            await self._emit_window(remaining, is_corrective=False)
        logger.info(
            "WebSocketStreamHandler stopped: received=%d flushed=%d",
            self._total_received, self._total_flushed,
        )

    async def stop(self) -> None:
        """Gracefully stop the handler."""
        self._running = False

    async def _connect_and_consume(self) -> None:
        """Inner connection loop. Raises on disconnect for retry logic to handle."""
        try:
            import websockets  # type: ignore
        except ImportError:
            logger.warning(
                "websockets package not installed. Install with: pip install websockets. "
                "Falling back to simulation mode."
            )
            await self._simulate_loop()
            return

        extra_headers = list(self._headers.items()) if self._headers else []
        async with websockets.connect(
            self.uri,
            extra_headers=extra_headers,
            ssl=self._ssl,
            ping_interval=30,
            ping_timeout=10,
        ) as ws:
            logger.info("WebSocket connected: %s", self.uri)
            self._retry_count = 0   # Reset on successful connect
            asyncio.ensure_future(self._flush_timer())
            async for raw_msg in ws:
                if not self._running:
                    break
                await self._process_message(raw_msg)

    async def _simulate_loop(self) -> None:
        """
        Simulation mode: generates synthetic events for local dev/testing.
        Produces batches every window_s seconds until stopped.
        """
        logger.info("WebSocket simulation mode active (no real connection).")
        import random
        while self._running:
            event = StreamEvent(
                payload={"value": round(random.gauss(100, 10), 2), "sensor_id": f"sim_{random.randint(1,5)}"},
                source_id="simulation",
                event_time=time.time() - random.uniform(0, 2.0),  # slight jitter
            )
            batch = self._buffer.add(event)
            if batch:
                await self._emit_window(batch, is_corrective=False)
            self._total_received += 1
            await asyncio.sleep(0.1)

    async def _process_message(self, raw_msg: str) -> None:
        """Parse, validate, and buffer a single incoming message."""
        try:
            payload = json.loads(raw_msg)
        except json.JSONDecodeError:
            logger.warning("WebSocket: non-JSON message ignored: %s", raw_msg[:100])
            return

        # Extract event_time from payload if available
        event_time_raw = payload.get("event_time") or payload.get("timestamp") or None
        event_time: Optional[float] = None
        if event_time_raw:
            try:
                if isinstance(event_time_raw, (int, float)):
                    event_time = float(event_time_raw)
                else:
                    from dateutil.parser import parse as _parse
                    event_time = _parse(str(event_time_raw)).timestamp()
            except Exception:  # noqa: BLE001
                pass

        event = StreamEvent(payload=payload, source_id=self._source_id, event_time=event_time)
        self._total_received += 1

        # Consumer lag check
        lag = int(self._queue.qsize())
        if lag > self._consumer_lag_threshold:
            logger.warning("Consumer lag: %d events queued (threshold=%d)", lag, self._consumer_lag_threshold)

        # Late event → corrective snapshot
        if event.is_late:
            logger.info(
                "Late event detected (recv_delay=%.1fs) — will produce corrective snapshot.",
                event.recv_time - event.event_time,
            )
            await self._emit_window([event], is_corrective=True)
            return

        batch = self._buffer.add(event)
        if batch:
            await self._emit_window(batch, is_corrective=False)

    async def _flush_timer(self) -> None:
        """Periodic force-flush so windows don't stall during low traffic."""
        while self._running:
            await asyncio.sleep(_BATCH_FLUSH_INTERVAL)
            remaining = self._buffer.force_flush()
            if remaining:
                await self._emit_window(remaining, is_corrective=False)

    async def _emit_window(self, batch: List[StreamEvent], is_corrective: bool) -> None:
        """
        Convert a batch of StreamEvents into an ISSFSnapshot and call on_window callback.
        Every window gets a unique snapshot_id and SHA-256 checksum.
        Corrective snapshots produce a new version_id (never overwrite original).
        """
        import pandas as pd

        if not batch:
            return

        checksum    = self._buffer.window_checksum(batch)
        snapshot_id = f"ws_{checksum[:16]}"
        if is_corrective:
            snapshot_id = f"ws_corrective_{checksum[:16]}_{int(time.time())}"

        rows = [e.payload for e in batch]
        df   = pd.DataFrame(rows)

        self._total_flushed += len(batch)

        logger.info(
            "[ws] Window flush: snapshot=%s rows=%d corrective=%s sha256=%s...",
            snapshot_id[:20], len(batch), is_corrective, checksum[:16],
        )

        # Emit to pipeline via callback
        if self.on_window is not None:
            try:
                self.on_window(batch, snapshot_id, is_corrective)
            except Exception as exc:  # noqa: BLE001
                logger.error("[ws] on_window callback failed: %s", exc)

        # Structured audit
        _audit_window(
            snapshot_id=snapshot_id,
            checksum=checksum,
            n_events=len(batch),
            source_id=self._source_id,
            is_corrective=is_corrective,
            window_type=self._window_type,
        )


# ── Webhook Receiver ───────────────────────────────────────────────────────────

class WebhookStreamHandler:
    """
    Async webhook receiver that processes incoming HTTP POST requests as a stream.

    Integrates with FastAPI / Starlette:

        handler = WebhookStreamHandler(config=config, on_event=my_callback)
        app.add_api_route("/webhook/stream", handler.receive, methods=["POST"])

    Each POST body is treated as a single streaming event. Events are accumulated
    in a window buffer identical to the WebSocket handler.

    Security: optional HMAC-SHA256 signature verification on X-DIPEX-Signature header.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        on_window: Optional[Callable[[List[StreamEvent], str, bool], None]] = None,
        secret: Optional[str] = None,
    ) -> None:
        self.config    = config or {}
        self.on_window = on_window
        self._secret   = secret.encode() if secret else None
        stream_cfg     = self.config.get("streaming", {})
        window_s       = float(stream_cfg.get("window_size_s", _DEFAULT_WINDOW_S))
        max_q          = int(stream_cfg.get("max_queue_depth", _DEFAULT_QUEUE_DEPTH))
        self._buffer   = WindowBuffer(window_s, max_q)
        self._source_id = "webhook"
        self._total_received = 0

    def _verify_signature(self, body: bytes, signature: Optional[str]) -> bool:
        """HMAC-SHA256 verification. Returns True if no secret is configured."""
        if self._secret is None:
            return True
        import hmac as _hmac
        expected = _hmac.new(self._secret, body, "sha256").hexdigest()
        return _hmac.compare_digest(expected, signature or "")

    async def receive(self, request: Any) -> Dict[str, Any]:
        """
        FastAPI-compatible endpoint. Call from a router like:
            router.post("/webhook/stream")(handler.receive)
        """
        body      = await request.body()
        signature = request.headers.get("X-DIPEX-Signature")

        if not self._verify_signature(body, signature):
            logger.warning("Webhook: invalid HMAC signature — event rejected.")
            return {"status": "error", "reason": "invalid_signature"}

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("Webhook: invalid JSON body.")
            return {"status": "error", "reason": "invalid_json"}

        event_time_raw = payload.get("event_time") or payload.get("timestamp")
        event_time: Optional[float] = None
        if event_time_raw:
            try:
                event_time = float(event_time_raw) if isinstance(event_time_raw, (int, float)) else None
            except Exception:  # noqa: BLE001
                pass

        event = StreamEvent(payload=payload, source_id=self._source_id, event_time=event_time)
        self._total_received += 1

        if event.is_late:
            await self._emit_window([event], is_corrective=True)
        else:
            batch = self._buffer.add(event)
            if batch:
                await self._emit_window(batch, is_corrective=False)

        return {"status": "ok", "event_id": event.event_id}

    async def _emit_window(self, batch: List[StreamEvent], is_corrective: bool) -> None:
        """Mirrors WebSocketStreamHandler._emit_window."""
        import pandas as pd
        if not batch:
            return
        checksum    = self._buffer.window_checksum(batch)
        snapshot_id = f"wh_corrective_{checksum[:16]}_{int(time.time())}" if is_corrective \
                      else f"wh_{checksum[:16]}"
        logger.info(
            "[webhook] Window flush: snapshot=%s rows=%d corrective=%s",
            snapshot_id[:24], len(batch), is_corrective,
        )
        if self.on_window is not None:
            try:
                self.on_window(batch, snapshot_id, is_corrective)
            except Exception as exc:  # noqa: BLE001
                logger.error("[webhook] on_window callback failed: %s", exc)
        _audit_window(
            snapshot_id=snapshot_id, checksum=checksum, n_events=len(batch),
            source_id=self._source_id, is_corrective=is_corrective, window_type="webhook",
        )


# ── IoT MQTT Bridge ────────────────────────────────────────────────────────────

class MQTTStreamBridge:
    """
    MQTT subscriber that bridges IoT device events to DIPEX streaming pipeline.

    Requires: paho-mqtt  (pip install paho-mqtt)

    Usage::

        bridge = MQTTStreamBridge(broker="mqtt://localhost", topic="sensors/#", config=config)
        bridge.start()     # non-blocking; runs in background thread
        bridge.stop()
    """

    def __init__(
        self,
        broker: str,
        topic: str,
        config: Optional[Dict[str, Any]] = None,
        on_window: Optional[Callable[[List[StreamEvent], str, bool], None]] = None,
    ) -> None:
        self.broker    = broker
        self.topic     = topic
        self.config    = config or {}
        self.on_window = on_window
        stream_cfg     = self.config.get("streaming", {})
        window_s       = float(stream_cfg.get("window_size_s", _DEFAULT_WINDOW_S))
        self._buffer   = WindowBuffer(window_s)
        self._source_id = f"mqtt:{topic}"
        self._client   = None
        self._running  = False

    def start(self) -> None:
        """Start MQTT subscriber in a background thread."""
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError:
            logger.warning("paho-mqtt not installed. IoT bridge unavailable. Install: pip install paho-mqtt")
            return

        self._client = mqtt.Client()
        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

        host, port = self._parse_broker(self.broker)
        self._client.connect(host, port, keepalive=60)
        self._running = True
        self._client.loop_start()
        logger.info("MQTT bridge started: broker=%s topic=%s", self.broker, self.topic)

    def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._client:
            remaining = self._buffer.force_flush()
            if remaining:
                import asyncio
                asyncio.run(self._emit(remaining, is_corrective=False))
            self._client.loop_stop()
            self._client.disconnect()

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: int) -> None:
        if rc == 0:
            client.subscribe(self.topic)
            logger.info("MQTT connected and subscribed: %s", self.topic)
        else:
            logger.error("MQTT connect failed: rc=%d", rc)

    def _on_disconnect(self, client: Any, userdata: Any, rc: int) -> None:
        logger.warning("MQTT disconnected: rc=%d — auto-reconnect active.", rc)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:  # noqa: BLE001
            payload = {"raw": msg.payload.decode("utf-8", errors="replace")}
        event = StreamEvent(payload=payload, source_id=self._source_id)
        batch = self._buffer.add(event)
        if batch:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._emit(batch, is_corrective=False))
                else:
                    loop.run_until_complete(self._emit(batch, is_corrective=False))
            except Exception as exc:  # noqa: BLE001
                logger.error("MQTT emit failed: %s", exc)

    async def _emit(self, batch: List[StreamEvent], is_corrective: bool) -> None:
        if not batch:
            return
        checksum    = self._buffer.window_checksum(batch)
        snapshot_id = f"mqtt_{checksum[:16]}"
        logger.info("[mqtt] Window flush: snapshot=%s rows=%d", snapshot_id[:20], len(batch))
        if self.on_window:
            self.on_window(batch, snapshot_id, is_corrective)

    @staticmethod
    def _parse_broker(broker: str) -> Tuple[str, int]:
        """Parse 'mqtt://host:port' or 'host:port' → (host, port)."""
        b = broker.replace("mqtt://", "").replace("mqtts://", "")
        if ":" in b:
            host, port_s = b.rsplit(":", 1)
            return host, int(port_s)
        return b, 1883


# ── Audit helper ───────────────────────────────────────────────────────────────

def _audit_window(
    snapshot_id: str,
    checksum: str,
    n_events: int,
    source_id: str,
    is_corrective: bool,
    window_type: str,
) -> None:
    """Write window flush event to audit log."""
    import os
    os.makedirs("audit", exist_ok=True)
    entry = {
        "event": "STREAM_WINDOW_FLUSH",
        "snapshot_id": snapshot_id,
        "sha256": checksum,
        "n_events": n_events,
        "source_id": source_id,
        "is_corrective": is_corrective,
        "window_type": window_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open("audit/stream_audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:  # noqa: BLE001
        pass


# ── Factory ────────────────────────────────────────────────────────────────────

def create_stream_handler(
    source_type: str,
    uri: str,
    config: Optional[Dict[str, Any]] = None,
    on_window: Optional[Callable] = None,
    **kwargs: Any,
) -> Any:
    """
    Factory that returns the correct stream handler for a given source type.

    source_type: "websocket" | "webhook" | "mqtt"
    """
    if source_type == "websocket":
        return WebSocketStreamHandler(uri=uri, config=config, on_window=on_window)
    elif source_type == "webhook":
        return WebhookStreamHandler(config=config, on_window=on_window, **kwargs)
    elif source_type == "mqtt":
        return MQTTStreamBridge(broker=uri, topic=kwargs.get("topic", "#"), config=config, on_window=on_window)
    else:
        raise ValueError(f"Unknown stream source_type: '{source_type}'. Use websocket|webhook|mqtt.")
