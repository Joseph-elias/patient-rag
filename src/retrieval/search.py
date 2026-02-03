from __future__ import annotations
import argparse
import json
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from src.retrieval.faiss_store import load_faiss_index


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def filter_candidates(items: List[Dict[str, Any]], patient_id: Optional[str]) -> np.ndarray:
    """
    Return indices of candidate chunks (row indices in items).
    If patient_id is None -> all.
    """
    if patient_id is None:
        return np.arange(len(items), dtype=np.int64)

    pid = patient_id.upper()
    idx = [i for i, it in enumerate(items) if str(it.get("patient_id", "")).upper() == pid]
    return np.array(idx, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Search FAISS index with optional patient_id filter")
    parser.add_argument("--index", type=str, default="indices/faiss/index.faiss")
    parser.add_argument("--chunks", type=str, default="data/processed/chunks.jsonl")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--patient_id", type=str, default=None, help="Optional patient filter, e.g., P001")
    args = parser.parse_args()

    items = read_jsonl(args.chunks)
    index = load_faiss_index(args.index)

    model = SentenceTransformer(args.model)

    q = args.question
    q_emb = model.encode([q], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)  # for cosine with inner product

    # If no filter: standard search
    if args.patient_id is None:
        scores, ids = index.search(q_emb, args.top_k)
        scores = scores[0]
        ids = ids[0]
        results = [(int(i), float(s)) for i, s in zip(ids, scores) if i != -1]
    else:
        # Patient filter: search more, then filter
        candidates = filter_candidates(items, args.patient_id)
        if len(candidates) == 0:
            print(f"❌ No chunks found for patient_id={args.patient_id}")
            return

        # Strategy: search a larger K then filter (simple + effective)
        big_k = min(len(items), max(args.top_k * 10, 50))
        scores, ids = index.search(q_emb, big_k)
        scores = scores[0]
        ids = ids[0]

        filtered = []
        cand_set = set(candidates.tolist())
        for i, s in zip(ids, scores):
            if int(i) in cand_set:
                filtered.append((int(i), float(s)))
            if len(filtered) >= args.top_k:
                break
        results = filtered

    print("\nQUESTION:", q)
    if args.patient_id:
        print("PATIENT FILTER:", args.patient_id.upper())

    for rank, (row_idx, score) in enumerate(results, 1):
        it = items[row_idx]
        print(f"\n--- Rank {rank} | score={score:.4f} | patient={it.get('patient_id')} | doc={it.get('doc_id')} | page={it.get('page')} | chunk_id={it.get('chunk_id')} ---")
        print(it["text"])


if __name__ == "__main__":
    main()
