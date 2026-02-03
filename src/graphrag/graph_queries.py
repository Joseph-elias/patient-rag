from typing import List
from src.graphrag.graph_store import PatientGraph


def get_adverse_events(graph: PatientGraph, patient_id: str) -> List[str]:
    p = graph.graph["patients"].get(patient_id, {})
    return [e["name"] for e in p.get("adverse_events", [])]


def get_negated_events(graph: PatientGraph, patient_id: str) -> List[str]:
    p = graph.graph["patients"].get(patient_id, {})
    return [n["name"] for n in p.get("negated_findings", [])]
