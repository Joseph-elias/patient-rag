from __future__ import annotations

import argparse
import json
import re
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.retrieval.faiss_store import load_faiss_index
from src.graphrag.graph_store import PatientGraph
from src.generation.prompting import build_system_prompt
from src.generation.llm_client import chat_completion
from src.generation.validate import validate_json_answer, ValidationError
from src.generation.guardrails import extract_focus_term, sources_contain_term


# =========================================================
# Utilities
# =========================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def strip_json_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


# =========================================================
# Retrieval
# =========================================================

def retrieve(
    *,
    question: str,
    items: List[Dict[str, Any]],
    index: faiss.Index,
    embed_model: SentenceTransformer,
    top_k: int = 5,
    patient_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    q_emb = embed_model.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)

    if patient_id is None:
        _, ids = index.search(q_emb, top_k)
        return [items[i] for i in ids[0].tolist() if i != -1]

    pid = patient_id.upper()
    candidates = [i for i, it in enumerate(items) if str(it.get("patient_id", "")).upper() == pid]
    if not candidates:
        return []

    cand_set = set(candidates)
    big_k = min(len(items), max(top_k * 10, 50))
    _, ids = index.search(q_emb, big_k)

    filtered = []
    for i in ids[0].tolist():
        if i == -1:
            continue
        if i in cand_set:
            filtered.append(items[i])
        if len(filtered) >= top_k:
            break
    return filtered


# =========================================================
# GraphRAG helpers
# =========================================================

def format_graph_facts(patient_facts: Optional[dict]) -> str:
    if not patient_facts:
        return "No structured patient facts available."

    lines = []

    if patient_facts.get("diagnosis"):
        lines.append(f"- Diagnosis: {patient_facts['diagnosis']['value']}")

    if patient_facts.get("treatment"):
        lines.append(f"- Treatment: {patient_facts['treatment']['value']}")

    if patient_facts.get("adverse_events"):
        evs = ", ".join(
            f"{e['name']} (grade {e['grade']})"
            for e in patient_facts["adverse_events"]
        )
        lines.append(f"- Known adverse events: {evs}")

    if patient_facts.get("negated_findings"):
        negs = ", ".join(n["name"] for n in patient_facts["negated_findings"])
        lines.append(f"- Known absent conditions: {negs}")

    if patient_facts.get("follow_up"):
        lines.append(f"- Follow-up: {patient_facts['follow_up']['value']}")

    return "\n".join(lines)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Ask questions with RAG + GraphRAG.")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--patient_id", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--chunks", type=str, default="data/processed/chunks.jsonl")
    parser.add_argument("--index", type=str, default="indices/faiss/index.faiss")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--provider", type=str, default="groq")
    parser.add_argument("--llm_model", type=str, default="llama-3.1-8b-instant")

    args = parser.parse_args()

    # Load data
    items = read_jsonl(args.chunks)
    index = load_faiss_index(args.index)
    embed_model = SentenceTransformer(args.embed_model)

    hits = retrieve(
        question=args.question,
        items=items,
        index=index,
        embed_model=embed_model,
        top_k=args.top_k,
        patient_id=args.patient_id,
    )

    if not hits:
        print("❌ No sources retrieved.")
        return

    # =========================================================
    # Guardrail: block hallucinated focus terms
    # =========================================================
    focus = extract_focus_term(args.question)
    if focus and not sources_contain_term(hits, focus):
        print("\n=== GUARDRAIL ===")
        print(f"Focus term '{focus}' not found in retrieved sources.")
        print("Returning empty structured answer.\n")

        empty = {
            "diagnosis": {"value": None, "evidence": None},
            "treatment": {"value": None, "evidence": None},
            "adverse_events": [],
            "negated_findings": [],
            "follow_up": {"value": None, "evidence": None},
            "other_notes": f"'{focus}' not mentioned in sources.",
        }

        print(json.dumps(empty, indent=2))
        return

    # =========================================================
    # GraphRAG: load patient facts
    # =========================================================
    graph = PatientGraph.load("data/graph/patient_graph.json")
    patient_facts = (
        graph.graph["patients"].get(args.patient_id)
        if args.patient_id
        else None
    )
    graph_context = format_graph_facts(patient_facts)

    # =========================================================
    # Build prompts (Graph + RAG)
    # =========================================================
    system_prompt = build_system_prompt()

    sources_text = "\n\n".join(
        f"[S{i+1}] {h.get('text', '')}" for i, h in enumerate(hits)
    )

    user_prompt = f"""
You are answering a clinical question using validated medical records.

STRUCTURED PATIENT FACTS (from knowledge graph — trusted):
{graph_context}

--------------------------------------------------
RETRIEVED SOURCE EXCERPTS:
{sources_text}

--------------------------------------------------
QUESTION:
{args.question}

IMPORTANT:
- Use structured patient facts when relevant.
- Do NOT contradict known absent conditions.
- Cite evidence using [S#].
- Output a single valid JSON object only.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # =========================================================
    # LLM call
    # =========================================================
    answer = chat_completion(
        messages=messages,
        model=args.llm_model,
        provider=args.provider,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("\n=== RAW MODEL OUTPUT ===\n")
    print(answer)

    print("\n=== PARSED JSON (validated) ===\n")
    try:
        answer_clean = strip_json_fence(answer)
        parsed = validate_json_answer(answer_clean, num_sources=len(hits))

        # Update graph
        graph.update_from_validated_json(args.patient_id, parsed)
        graph.save("data/graph/patient_graph.json")

        print(json.dumps(parsed, indent=2))

    except ValidationError as e:
        print(f"❌ Validation failed: {e}")

    print("\n=== SOURCES (verbatim chunks) ===\n")
    for i, h in enumerate(hits, start=1):
        print(
            f"[S{i}] doc={h.get('doc_id')} "
            f"page={h.get('page')} "
            f"chunk_id={h.get('chunk_id')} "
            f"patient={h.get('patient_id')}"
        )
        print(h.get("text", ""))
        print("-" * 80)


if __name__ == "__main__":
    main()
