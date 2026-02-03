from __future__ import annotations
from typing import List, Dict


def build_context_block(hits: List[Dict]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        citation = f"[S{i}: doc={h.get('doc_id')} page={h.get('page')} chunk_id={h.get('chunk_id')}]"
        text = (h.get("text") or "").strip()
        parts.append(f"{citation}\n{text}")
    return "\n\n".join(parts)


def build_system_prompt() -> str:
    return (
        "You are a clinical information extraction assistant.\n\n"
        "You MUST ALWAYS output a SINGLE JSON object with EXACTLY the following keys:\n"
        "- diagnosis\n"
        "- treatment\n"
        "- adverse_events\n"
        "- negated_findings\n"
        "- follow_up\n"
        "- other_notes\n\n"
        "Schema rules:\n"
        "- diagnosis, treatment, follow_up MUST be objects: "
        '{ "value": string|null, "evidence": "[S#]"|null }\n'
        "- adverse_events MUST be a list of objects: "
        '{ "name": string, "grade": int|null, "evidence": "[S#]" }\n'
        "- negated_findings MUST be a list of objects: "
        '{ "name": string, "evidence": "[S#]" }\n'
        "- other_notes MUST be string or null\n\n"
        "If information is not present, set value to null and lists to empty.\n"
        "DO NOT omit keys.\n"
        "DO NOT add new keys.\n"
        "DO NOT change the schema.\n"
        "DO NOT output markdown or explanations.\n"
        "Return JSON ONLY."
    )


def build_user_prompt(question: str, hits: List[Dict]) -> str:
    context = build_context_block(hits)

    return (
        f"QUESTION:\n{question}\n\n"
        "Use ONLY the SOURCES below.\n"
        "Extract structured medical facts.\n"
        "Every extracted field MUST include an evidence tag like [S1], [S2], etc.\n"
        "If a field is not supported by the sources, set it to null / empty.\n\n"
        f"SOURCES:\n{context}\n\n"
        "Remember: you MUST return the FULL JSON schema with ALL keys."
    )
