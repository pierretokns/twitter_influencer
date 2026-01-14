# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "opentelemetry-api>=1.20.0",
#     "opentelemetry-sdk>=1.20.0",
# ]
# ///
"""
SQLite OpenTelemetry Exporter
=============================

Self-hosted OpenTelemetry tracing with SQLite storage.
No cloud service required - all spans stored locally.

USAGE:
    from agents.telemetry import setup_telemetry, get_tracer

    # Initialize once at startup
    setup_telemetry()

    # Get tracer and create spans
    tracer = get_tracer()
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("key", "value")
        # ... do work ...

SCHEMA:
    Spans are stored in the otel_spans table with:
    - span_id, trace_id, parent_span_id
    - name, kind, status_code, status_message
    - start_time_unix_nano, end_time_unix_nano, duration_ms (computed)
    - attributes, events, resource (JSON)
"""

import json
import sqlite3
import threading
from typing import Optional, Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME


# Thread-local storage for database connections
_local = threading.local()

# Global tracer provider
_tracer_provider: Optional[TracerProvider] = None


class SQLiteSpanExporter(SpanExporter):
    """
    Custom OpenTelemetry exporter that writes spans to SQLite.

    Implements the SpanExporter interface for seamless OTEL integration.
    Spans are batched and written to the otel_spans table.
    """

    def __init__(self, db_path: str):
        """
        Initialize the SQLite exporter.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(_local, 'conn') or _local.conn is None:
            _local.conn = sqlite3.connect(self.db_path)
            _local.conn.row_factory = sqlite3.Row
        return _local.conn

    def _ensure_schema(self):
        """Ensure the otel_spans table exists."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS otel_spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                parent_span_id TEXT,
                name TEXT NOT NULL,
                kind TEXT DEFAULT 'INTERNAL',
                start_time_unix_nano INTEGER,
                end_time_unix_nano INTEGER,
                duration_ms REAL GENERATED ALWAYS AS
                    ((end_time_unix_nano - start_time_unix_nano) / 1000000.0) STORED,
                status_code TEXT DEFAULT 'OK',
                status_message TEXT,
                attributes JSON,
                events JSON,
                resource JSON,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_trace ON otel_spans(trace_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_name ON otel_spans(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_start ON otel_spans(start_time_unix_nano)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_parent ON otel_spans(parent_span_id)")
        conn.commit()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export a batch of spans to SQLite.

        Args:
            spans: Sequence of ReadableSpan objects to export

        Returns:
            SpanExportResult.SUCCESS or SpanExportResult.FAILURE
        """
        if not spans:
            return SpanExportResult.SUCCESS

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            for span in spans:
                # Extract span context
                ctx = span.get_span_context()
                span_id = format(ctx.span_id, '016x')
                trace_id = format(ctx.trace_id, '032x')

                # Get parent span ID if exists
                parent_span_id = None
                if span.parent and span.parent.span_id:
                    parent_span_id = format(span.parent.span_id, '016x')

                # Convert span kind to string
                kind = span.kind.name if span.kind else 'INTERNAL'

                # Convert status
                status_code = span.status.status_code.name if span.status else 'UNSET'
                status_message = span.status.description if span.status else None

                # Serialize attributes as JSON
                attributes = {}
                if span.attributes:
                    for key, value in span.attributes.items():
                        # Convert non-JSON-serializable types
                        if isinstance(value, (list, tuple)):
                            attributes[key] = list(value)
                        else:
                            attributes[key] = value

                # Serialize events as JSON array
                events = []
                if span.events:
                    for event in span.events:
                        event_dict = {
                            "name": event.name,
                            "timestamp": event.timestamp,
                            "attributes": dict(event.attributes) if event.attributes else {}
                        }
                        events.append(event_dict)

                # Serialize resource as JSON
                resource = {}
                if span.resource:
                    for key, value in span.resource.attributes.items():
                        resource[key] = value

                cursor.execute("""
                    INSERT OR REPLACE INTO otel_spans (
                        span_id, trace_id, parent_span_id, name, kind,
                        start_time_unix_nano, end_time_unix_nano,
                        status_code, status_message,
                        attributes, events, resource
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    span_id,
                    trace_id,
                    parent_span_id,
                    span.name,
                    kind,
                    span.start_time,
                    span.end_time,
                    status_code,
                    status_message,
                    json.dumps(attributes),
                    json.dumps(events),
                    json.dumps(resource),
                ))

            conn.commit()
            return SpanExportResult.SUCCESS

        except Exception as e:
            print(f"[SQLiteSpanExporter] Export failed: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        """Clean up resources on shutdown."""
        if hasattr(_local, 'conn') and _local.conn:
            _local.conn.close()
            _local.conn = None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush pending spans.

        For SQLite, writes are synchronous so this is a no-op.

        Args:
            timeout_millis: Timeout in milliseconds (unused)

        Returns:
            Always True
        """
        return True


def setup_telemetry(
    service_name: str = "linkedin_agents",
    db_path: str = "output_data/ai_news.db"
) -> TracerProvider:
    """
    Configure OpenTelemetry with SQLite exporter.

    Should be called once at application startup.

    Args:
        service_name: Name to identify this service in traces
        db_path: Path to the SQLite database

    Returns:
        Configured TracerProvider
    """
    global _tracer_provider

    if _tracer_provider is not None:
        return _tracer_provider

    # Create resource with service name
    resource = Resource(attributes={
        SERVICE_NAME: service_name,
    })

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Create and register the SQLite exporter
    exporter = SQLiteSpanExporter(db_path)
    processor = BatchSpanProcessor(exporter)
    _tracer_provider.add_span_processor(processor)

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    print(f"[Telemetry] Initialized with SQLite exporter: {db_path}")
    return _tracer_provider


def get_tracer(name: str = "agents") -> trace.Tracer:
    """
    Get a tracer for creating spans.

    If telemetry hasn't been set up, returns a no-op tracer.

    Args:
        name: Name for the tracer (used for identifying instrumentation)

    Returns:
        OpenTelemetry Tracer
    """
    return trace.get_tracer(name)


def shutdown_telemetry():
    """
    Shutdown telemetry and flush any remaining spans.

    Should be called when the application is shutting down.
    """
    global _tracer_provider

    if _tracer_provider:
        _tracer_provider.shutdown()
        _tracer_provider = None


__all__ = [
    'SQLiteSpanExporter',
    'setup_telemetry',
    'get_tracer',
    'shutdown_telemetry',
]


if __name__ == "__main__":
    import time

    # Quick test
    print("Testing SQLite OTEL exporter...")

    setup_telemetry(db_path=":memory:")
    tracer = get_tracer("test")

    with tracer.start_as_current_span("test_parent") as parent_span:
        parent_span.set_attribute("test.attribute", "hello")
        parent_span.set_attribute("test.number", 42)

        time.sleep(0.1)

        with tracer.start_as_current_span("test_child") as child_span:
            child_span.set_attribute("child.data", "world")
            time.sleep(0.05)

    # Force flush
    shutdown_telemetry()

    print("Test complete - spans exported to in-memory database")
