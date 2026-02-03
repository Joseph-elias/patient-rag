from __future__ import annotations

RAG_JSON_SCHEMA_HINT = """
Return ONLY valid JSON. No markdown. No extra text.

JSON format:
{
  "diagnosis": string|null,
  "treatment": string|null,
  "adverse_events": [
    {"name": string, "grade": number|null, "evidence": "[S1]"|"[S2]"|...}
  ],
  "negated_findings": [
    {"name": string, "evidence": "[S1]"|"[S2]"|...}
  ],
  "follow_up": string|null,
  "other_notes": string|null
}

Rules:
- Use ONLY the sources.
- Every extracted field MUST include evidence citations like "[S1]".
- Put explicitly negated events (e.g., "No evidence of X") in "negated_findings".
- Put response/progression/stable disease in "follow_up", not adverse_events.
""".strip()
