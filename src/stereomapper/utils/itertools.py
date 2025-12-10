"""Iterator utilities for batch processing."""

from typing import Iterable, Sequence, TypeVar

T = TypeVar('T')

def chunked(seq: Sequence[T] | Iterable[T], size: int) -> Iterable[list[T]]:
    """
    Yield lists of length `size` (last chunk may be smaller).
    
    Args:
        seq: Input sequence or iterable
        size: Maximum size of each chunk
        
    Yields:
        Lists of items from the input sequence
    """
    batch: list[T] = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch