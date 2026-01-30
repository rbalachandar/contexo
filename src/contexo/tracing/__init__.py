"""OpenTelemetry tracing support for Contexo.

This module provides tracing instrumentation for Contexo operations,
allowing you to observe memory operations, search queries, and
context retrieval in your observability platform.

Example:
    ```python
    from contexo.tracing import setup_tracing
    from contexo import Contexo, local_config

    # Enable tracing with console export (for development)
    setup_tracing(service_name="my-app")

    # Or export to OTLP endpoint (for production)
    setup_tracing(
        service_name="my-app",
        endpoint="http://localhost:4317",
    )

    ctx = Contexo(config=local_config())
    await ctx.initialize()

    # All operations are now traced automatically
    await ctx.add_message("Hello!", role="user")
    results = await ctx.search_memory("greeting")
    ```
"""

from __future__ import annotations

from contexo.tracing.decorator import (
    trace_async_method,
    trace_context_manager,
    trace_method,
)
from contexo.tracing.tracing import get_tracer, is_tracing_enabled, setup_tracing

__all__ = [
    "setup_tracing",
    "get_tracer",
    "is_tracing_enabled",
    "trace_method",
    "trace_async_method",
    "trace_context_manager",
]
