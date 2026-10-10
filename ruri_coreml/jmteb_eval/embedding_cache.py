"""Chunked on-disk embedding cache, so a run interrupted at the end of the
night window resumes from the last finished chunk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .constants import CACHE_CHUNK_SIZE

_CHUNK_FILE_TEMPLATE = "chunk_{index:05d}.npy"
_COMPLETE_FILE_NAME = "embeddings.npy"


def cached_embeddings(
    cache_directory: Path,
    texts: list[str],
    encode: Callable[[list[str]], np.ndarray],
) -> np.ndarray:
    complete_path = cache_directory / _COMPLETE_FILE_NAME
    if complete_path.exists():
        return np.load(complete_path)

    cache_directory.mkdir(parents=True, exist_ok=True)
    chunks = []
    for index, start in enumerate(range(0, len(texts), CACHE_CHUNK_SIZE)):
        chunk_path = cache_directory / _CHUNK_FILE_TEMPLATE.format(index=index)
        if not chunk_path.exists():
            np.save(chunk_path, encode(texts[start : start + CACHE_CHUNK_SIZE]))
        chunks.append(np.load(chunk_path))

    embeddings = np.concatenate(chunks)
    np.save(complete_path, embeddings)
    for chunk_path in cache_directory.glob("chunk_*.npy"):
        chunk_path.unlink()
    return embeddings
