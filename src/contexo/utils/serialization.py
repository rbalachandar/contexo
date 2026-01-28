"""Serialization utilities for memory entries and related types."""

from __future__ import annotations

import json
from base64 import b64decode, b64encode
from typing import Any

from contexo.core.memory import EntryType, MemoryEntry


def serialize_entry(entry: MemoryEntry) -> dict[str, Any]:
    """Serialize a MemoryEntry to a dictionary.

    Args:
        entry: The entry to serialize

    Returns:
        Dictionary representation of the entry
    """
    return {
        "id": entry.id,
        "entry_type": entry.entry_type.value,
        "content": entry.content,
        "metadata": entry.metadata,
        "timestamp": entry.timestamp,
        "token_count": entry.token_count,
        "importance_score": entry.importance_score,
        "embedding": _serialize_embedding(entry.embedding) if entry.embedding else None,
        "parent_id": entry.parent_id,
        "conversation_id": entry.conversation_id,
    }


def deserialize_entry(data: dict[str, Any]) -> MemoryEntry:
    """Deserialize a dictionary to a MemoryEntry.

    Args:
        data: Dictionary representation of the entry

    Returns:
        A MemoryEntry instance
    """
    return MemoryEntry(
        id=data["id"],
        entry_type=EntryType(data["entry_type"]),
        content=data["content"],
        metadata=data.get("metadata", {}),
        timestamp=data.get("timestamp", 0.0),
        token_count=data.get("token_count", 0),
        importance_score=data.get("importance_score", 0.5),
        embedding=_deserialize_embedding(data.get("embedding")),
        parent_id=data.get("parent_id"),
        conversation_id=data.get("conversation_id"),
    )


def serialize_entries(entries: list[MemoryEntry]) -> list[dict[str, Any]]:
    """Serialize multiple MemoryEntry objects.

    Args:
        entries: List of entries to serialize

    Returns:
        List of dictionary representations
    """
    return [serialize_entry(entry) for entry in entries]


def deserialize_entries(data: list[dict[str, Any]]) -> list[MemoryEntry]:
    """Deserialize multiple MemoryEntry objects.

    Args:
        data: List of dictionary representations

    Returns:
        List of MemoryEntry instances
    """
    return [deserialize_entry(entry) for entry in data]


def entry_to_json(entry: MemoryEntry, indent: int | None = None) -> str:
    """Convert a MemoryEntry to a JSON string.

    Args:
        entry: The entry to convert
        indent: JSON indentation level

    Returns:
        JSON string representation
    """
    return json.dumps(serialize_entry(entry), indent=indent)


def entry_from_json(json_str: str) -> MemoryEntry:
    """Parse a MemoryEntry from a JSON string.

    Args:
        json_str: JSON string representation

    Returns:
        A MemoryEntry instance
    """
    data = json.loads(json_str)
    return deserialize_entry(data)


def _serialize_embedding(embedding: list[float]) -> str:
    """Serialize an embedding vector to base64 string.

    Args:
        embedding: The embedding vector

    Returns:
        Base64 encoded string
    """
    import struct

    # Pack floats as bytes and encode as base64
    bytes_data = struct.pack(f"{len(embedding)}f", *embedding)
    return b64encode(bytes_data).decode("utf-8")


def _deserialize_embedding(data: str | None) -> list[float] | None:
    """Deserialize a base64 string to an embedding vector.

    Args:
        data: Base64 encoded string

    Returns:
        The embedding vector, or None if data is None
    """
    if data is None:
        return None

    import struct

    bytes_data = b64decode(data.encode("utf-8"))
    # Unpack bytes to floats
    float_count = len(bytes_data) // 4
    return list(struct.unpack(f"{float_count}f", bytes_data))


def export_context(
    entries: list[MemoryEntry],
    format: str = "json",
) -> str:
    """Export context entries to a specific format.

    Args:
        entries: List of entries to export
        format: Export format ("json", "txt", "markdown")

    Returns:
        Formatted string

    Raises:
        ValueError: If format is unknown
    """
    format = format.lower()

    if format == "json":
        return json.dumps(serialize_entries(entries), indent=2)

    if format == "txt":
        lines = []
        for entry in entries:
            lines.append(f"[{entry.entry_type.value}] {entry.content}")
        return "\n".join(lines)

    if format in ("markdown", "md"):
        lines = []
        for entry in entries:
            role = entry.metadata.get("role", entry.entry_type.value)
            lines.append(f"**{role}**: {entry.content}")
        return "\n\n".join(lines)

    raise ValueError(f"Unknown export format: {format}")
