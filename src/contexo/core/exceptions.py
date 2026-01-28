"""Custom exceptions for the Contexo library."""


class ContexoError(Exception):
    """Base exception for all Contexo errors."""

    pass


class ConfigurationError(ContexoError):
    """Raised when there is an error in configuration."""

    pass


class StorageError(ContexoError):
    """Raised when there is an error with storage backend."""

    pass


class EmbeddingError(ContexoError):
    """Raised when there is an error with embedding generation."""

    pass


class TokenLimitError(ContexoError):
    """Raised when token limits cannot be satisfied."""

    pass


class CompactionError(ContexoError):
    """Raised when context compaction fails."""

    pass


class SearchError(ContexoError):
    """Raised when a search operation fails."""

    pass


class ValidationError(ContexoError):
    """Raised when input validation fails."""

    pass
