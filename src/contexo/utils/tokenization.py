"""Token counting utilities."""

from __future__ import annotations

import re
from typing import Callable


def count_tokens_by_words(text: str) -> int:
    """Count tokens by approximating with words.

    This is a simple approximation that counts words and punctuation.
    It's not as accurate as using a proper tokenizer but works for
    basic estimation.

    Args:
        text: The text to count tokens for

    Returns:
        Estimated token count
    """
    # Split by whitespace and count
    words = text.split()
    # Add tokens for punctuation (rough approximation)
    punctuation_count = len(re.findall(r'[^\w\s]', text))
    return len(words) + (punctuation_count // 2)


def count_tokens_by_characters(text: str, chars_per_token: int = 4) -> int:
    """Count tokens by character approximation.

    Args:
        text: The text to count tokens for
        chars_per_token: Average characters per token (default 4)

    Returns:
        Estimated token count
    """
    return max(1, len(text) // chars_per_token)


def count_tokens_by_tiktoken(text: str, model: str = "gpt-4") -> int:
    """Count tokens using tiktoken tokenizer.

    This provides accurate token counts for OpenAI models.

    Args:
        text: The text to count tokens for
        model: The model name to use for tokenization

    Returns:
        Accurate token count

    Raises:
        ImportError: If tiktoken is not installed
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError("tiktoken is required for accurate token counting. Install with: pip install tiktoken")

    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def create_token_counter(method: str = "words", **kwargs: str | int) -> Callable[[str], int]:
    """Create a token counter function.

    Args:
        method: The counting method ("words", "chars", "tiktoken")
        **kwargs: Additional arguments for the method

    Returns:
        A function that counts tokens

    Raises:
        ValueError: If the method is unknown
    """
    method = method.lower()

    if method == "words":
        return count_tokens_by_words

    if method == "chars":
        chars_per_token = kwargs.get("chars_per_token", 4)
        return lambda text: count_tokens_by_characters(text, int(chars_per_token))  # type: ignore

    if method == "tiktoken":
        model = kwargs.get("model", "gpt-4")
        return lambda text: count_tokens_by_tiktoken(text, str(model))

    raise ValueError(f"Unknown token counting method: {method}")


# Default token counter using word approximation
default_token_counter = count_tokens_by_words
