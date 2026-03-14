import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StreamWindowConfig:
    """Configuration for event-time windowing and watermark behaviour."""
    watermark_seconds: int = 300  # allowed lateness in seconds
    window_seconds: int = 60      # tumbling window size


class StreamLoader:
    """
    Handles streaming data ingestion (Kafka, Webhooks, API Feeds).

    This loader models:
      - event-time timestamps (expects a 'timestamp' field in events)
      - watermark-based lateness handling
      - tumbling windows over the event-time axis
    """

    def __init__(
        self,
        bootstrap_servers: str = "kafka:29092",
        mock_mode: bool = True,
        window_config: Optional[StreamWindowConfig] = None,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.mock_mode = mock_mode
        self._stop_event = threading.Event()
        self._window_cfg = window_config or StreamWindowConfig()

    def consume_kafka(
        self,
        topic: str,
        callback: Callable[[pd.DataFrame], None],
    ):
        """
        Consumes messages from a Kafka topic and triggers a callback with
        windowed DataFrames ordered by event-time.
        """
        if self.mock_mode:
            logger.info("StreamLoader: mocking Kafka consumption for topic=%s", topic)
            self._mock_consumer(topic, callback)
        else:
            # Placeholder for actual Kafka implementation with watermark & windows.
            # from kafka import KafkaConsumer
            # consumer = KafkaConsumer(topic, bootstrap_servers=self.bootstrap_servers)
            # self._consume_with_windows(consumer, callback)
            logger.warning(
                "Kafka consumer not implemented in live mode. Falling back to mock."
            )
            self._mock_consumer(topic, callback)

    # ------------------------------------------------------------------
    # Mock streaming with event-time windows
    # ------------------------------------------------------------------

    def _mock_consumer(
        self,
        topic: str,
        callback: Callable[[pd.DataFrame], None],
    ):
        """
        Simulates a stream of data for testing with event-time windows.
        """
        base_ts = time.time()
        mock_data = [
            {"id": 1, "value": 100, "timestamp": base_ts},
            {"id": 2, "value": 150, "timestamp": base_ts + 10},
            {"id": 3, "value": 120, "timestamp": base_ts + 65},  # next window
        ]

        buffer = []
        window_start = None

        for item in mock_data:
            if self._stop_event.is_set():
                break
            buffer.append(item)

            event_time = item["timestamp"]
            if window_start is None:
                window_start = event_time

            # When event_time exceeds current window, flush window buffer
            if event_time - window_start >= self._window_cfg.window_seconds:
                df = pd.DataFrame(buffer)
                df["event_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df = df.sort_values("event_time")
                callback(df)
                buffer = []
                window_start = event_time

            time.sleep(0.5)  # Simulate real-time delay

        # Flush any remaining events as a final window
        if buffer and not self._stop_event.is_set():
            df = pd.DataFrame(buffer)
            df["event_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df = df.sort_values("event_time")
            callback(df)

    def stop(self):
        """Stops any active stream consumption."""
        self._stop_event.set()
