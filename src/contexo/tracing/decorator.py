"""Tracing decorators for instrumenting Contexo operations.

This module provides decorators for automatically creating OpenTelemetry spans
around function calls, with support for both synchronous and async functions.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from contexo.tracing import get_tracer, is_tracing_enabled

P = ParamSpec("P")
T = TypeVar("T")


def trace_method(
    span_name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for tracing synchronous methods.

    Args:
        span_name: Name for the span. If None, uses the method name.
        attributes: Static attributes to add to all spans.

    Returns:
        Decorated function that creates a span when called.

    Example:
        ```python
        class MyClass:
            @trace_method("my.operation", {"static": "value"})
            def my_method(self, arg):
                return arg * 2
        ```
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not is_tracing_enabled():
                return func(*args, **kwargs)

            tracer = get_tracer()
            if tracer is None:
                return func(*args, **kwargs)

            name = span_name or f"{func.__module__}.{func.__name__}"
            attrs = dict(attributes) if attributes else {}

            with tracer.start_as_current_span(name) as span:
                # Add attributes from kwargs
                for key, value in kwargs.items():
                    if value is not None and not key.startswith("_"):
                        attrs[key] = (
                            str(value) if not isinstance(value, (int, float, bool)) else value
                        )

                # Add instance attributes if available (for methods)
                if args and hasattr(args[0], "__dict__"):
                    instance = args[0]
                    # Add common Contexo attributes
                    if hasattr(instance, "conversation_id"):
                        conv_id = instance.conversation_id
                        if conv_id:
                            attrs["conversation_id"] = conv_id

                for key, value in attrs.items():
                    span.set_attribute(key, value)

                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_async_method(
    span_name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for tracing async methods.

    Args:
        span_name: Name for the span. If None, uses the method name.
        attributes: Static attributes to add to all spans.

    Returns:
        Decorated async function that creates a span when called.

    Example:
        ```python
        class MyClass:
            @trace_async_method("my.async_operation")
            async def my_async_method(self, arg):
                return await process(arg)
        ```
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not is_tracing_enabled():
                return await func(*args, **kwargs)

            tracer = get_tracer()
            if tracer is None:
                return await func(*args, **kwargs)

            name = span_name or f"{func.__module__}.{func.__name__}"
            attrs = dict(attributes) if attributes else {}

            with tracer.start_as_current_span(name) as span:
                # Add attributes from kwargs
                for key, value in kwargs.items():
                    if value is not None and not key.startswith("_"):
                        attrs[key] = (
                            str(value) if not isinstance(value, (int, float, bool)) else value
                        )

                # Add instance attributes if available (for methods)
                if args and hasattr(args[0], "__dict__"):
                    instance = args[0]
                    if hasattr(instance, "conversation_id"):
                        conv_id = instance.conversation_id
                        if conv_id:
                            attrs["conversation_id"] = conv_id

                for key, value in attrs.items():
                    span.set_attribute(key, value)

                result = await func(*args, **kwargs)

                # Add result attributes for common return types
                if hasattr(result, "__len__"):
                    span.set_attribute("result.count", len(result))

                return result

        return wrapper

    return decorator


def trace_context_manager(
    span_name: str,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for tracing async context managers.

    This decorator is designed for context manager classes that implement
    __aenter__ and __aexit__, creating spans for the duration of the context.

    Args:
        span_name: Name for the span.
        attributes: Static attributes to add to the span.

    Returns:
        Decorated class with traced __aenter__ and __aexit__.

    Example:
        ```python
        @trace_context_manager("database.transaction")
        class DatabaseTransaction:
            async def __aenter__(self):
                await self.begin()
                return self

            async def __aexit__(self, *args):
                await self.commit()
        ```
    """

    def decorator(cls: type[T]) -> type[T]:
        if not is_tracing_enabled():
            return cls

        original_aenter = cls.__aenter__
        original_aexit = cls.__aexit__

        @functools.wraps(original_aenter)
        async def traced_aenter(self, *args, **kwargs):
            tracer = get_tracer()
            if tracer is None:
                return await original_aenter(self, *args, **kwargs)

            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                self._tracing_span = span
                return await original_aenter(self, *args, **kwargs)

        @functools.wraps(original_aexit)
        async def traced_aexit(self, *args, **kwargs):
            result = await original_aexit(self, *args, **kwargs)

            if hasattr(self, "_tracing_span"):
                self._tracing_span.end()
                delattr(self, "_tracing_span")

            return result

        cls.__aenter__ = traced_aenter
        cls.__aexit__ = traced_aexit
        return cls

    return decorator


__all__ = [
    "trace_method",
    "trace_async_method",
    "trace_context_manager",
]
