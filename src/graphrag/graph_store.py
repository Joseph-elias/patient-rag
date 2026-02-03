from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any


class PatientGraph:
    def __init__(self):
        self.graph: Dict[str, Any] = {"patients": {}}

    def add_patient(self, patient_id: str):
        if patient_id not in self.graph["patients"]:
            self.graph["patients"][patient_id] = {
                "diagnosis": None,
                "treatment": None,
                "adverse_events": [],
                "negated_findings": [],
                "follow_up": None,
            }

    def update_from_validated_json(self, patient_id: str, data: Dict[str, Any]):
        self.add_patient(patient_id)

        p = self.graph["patients"][patient_id]

        if data.get("diagnosis", {}).get("value"):
            p["diagnosis"] = data["diagnosis"]

        if data.get("treatment", {}).get("value"):
            p["treatment"] = data["treatment"]

        for ev in data.get("adverse_events", []):
            if ev not in p["adverse_events"]:
                p["adverse_events"].append(ev)

        for neg in data.get("negated_findings", []):
            if neg not in p["negated_findings"]:
                p["negated_findings"].append(neg)

        if data.get("follow_up", {}).get("value"):
            p["follow_up"] = data["follow_up"]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PatientGraph":
        g = cls()
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                g.graph = json.load(f)
        return g
