"""Validate binary node references before resolving them."""

from collections.abc import Sequence
from typing import TypeVar

from .basereader import EDMFormatError

T = TypeVar("T")


def resolve_node_indices(nodes: Sequence[T], indices: Sequence[int]) -> list[T]:
    """Resolve a skin palette only after every index has been validated."""
    for index in indices:
        if not isinstance(index, int) or not 0 <= index < len(nodes):
            raise EDMFormatError(
                f"Invalid bone index {index!r}; node count {len(nodes)}"
            )
    return [nodes[index] for index in indices]
