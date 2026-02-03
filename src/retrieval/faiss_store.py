from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import json

import numpy as np
import faiss


@dataclass
class FaissArtifacts:
    index_path: Path
    meta_path: Path


def save_faiss_index(index: faiss.Index, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def save_metadata(metadata: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cosine_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index for cosine similarity using inner product.
    We L2-normalize embeddings first, then cosine(u,v)=u·v.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product
    index.add(embeddings)
    return index
