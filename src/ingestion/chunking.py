from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: int
    text: str
    start_char: int
    end_char: int


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[Chunk]:
    """
    Simple character-based chunking.
    - chunk_size and overlap are in characters.
    - robust and easy to reason about.
    """
    text = " ".join(text.split())
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_str = text[start:end].strip()

        if chunk_str:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_str,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_id += 1

        if end == len(text):
            break

        start = max(end - overlap, 0)

    return chunks
