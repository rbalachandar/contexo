"""OpenTelemetry tracing setup and management for Contexo."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer = None
_tracer_provider: TracerProvider | None = None


def setup_tracing(
    service_name: str = "contexo",
    endpoint: str | None = None,
    console: bool = False,
    headers: dict[str, str] | None = None,
) -> None:
    """Set up OpenTelemetry tracing for Contexo.

    This function configures OpenTelemetry tracing for all Contexo operations.
    Call this once at application startup before creating any Contexo instances.

    Args:
        service_name: Name of your service for trace identification
        endpoint: OTLP endpoint (e.g., "http://localhost:4317" for OTLP gRPC)
                  If None, uses console exporter when console=True
        console: If True, export traces to console (useful for development)
        headers: Optional headers for OTLP exporter (e.g., for authentication)

    Example:
        ```python
        # Development: print traces to console
        setup_tracing(service_name="my-app", console=True)

        # Production: send to OTLP collector
        setup_tracing(
            service_name="my-app",
            endpoint="http://otel-collector:4317",
            headers={"Authorization": "Bearer token"}
        )
        ```

    Note:
        If neither `endpoint` nor `console` is specified, tracing is enabled
        but no exporter is configured. You can configure exporters manually
        using the returned TracerProvider.
    """
    global _tracer, _tracer_provider

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        # Create resource with service name
        resource = Resource.create({"service.name": service_name})

        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(_tracer_provider)

        # Configure exporters
        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            otlp_exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
            _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"Contexo tracing enabled: OTLP endpoint={endpoint}")
        elif console:
            _tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            logger.info("Contexo tracing enabled: console export")
        else:
            logger.warning("Contexo tracing enabled but no exporter configured")

        # Get tracer
        _tracer = trace.get_tracer(__name__)

    except ImportError:
        logger.warning(
            "OpenTelemetry not installed. Install with: pip install contexo[opentelemetry]"
        )


def get_tracer():
    """Get the configured OpenTelemetry tracer.

    Returns:
        The OpenTelemetry tracer instance, or None if tracing not configured.

    Example:
        ```python
        from contexo.tracing import get_tracer

        tracer = get_tracer()
        if tracer:
            with tracer.start_as_current_span("custom-operation"):
                # Your custom code here
        ```
    """
    return _tracer


def is_tracing_enabled() -> bool:
    """Check if OpenTelemetry tracing is enabled.

    Returns:
        True if tracing has been configured, False otherwise.
    """
    return _tracer is not None
