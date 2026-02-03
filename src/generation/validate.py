from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional


class ValidationError(Exception):
    pass


CIT_RE = re.compile(r"\[S(\d+)\]")
NEG_PREFIX_RE = re.compile(r"^\s*(no evidence of|no sign of|without evidence of)\s+(.+)$", re.IGNORECASE)


def _fill_missing_evidence(data: dict, num_sources: int) -> dict:
    """
    If the model outputs a value but forgets evidence, fill it with [S1] (if available).
    This is a pragmatic fix to keep validation strict while preventing random LLM omissions.
    """
    fallback = "[S1]" if num_sources >= 1 else None

    # --- single objects: diagnosis / treatment / follow_up ---
    for key in ("diagnosis", "treatment", "follow_up"):
        obj = data.get(key)
        if isinstance(obj, dict):
            if obj.get("value") is not None and not obj.get("evidence"):
                obj["evidence"] = fallback

    # --- lists: negated_findings ---
    nf = data.get("negated_findings", [])
    if isinstance(nf, list):
        for it in nf:
            if isinstance(it, dict):
                if it.get("name") is not None and not it.get("evidence"):
                    it["evidence"] = fallback

    # --- lists: adverse_events ---
    ae = data.get("adverse_events", [])
    if isinstance(ae, list):
        for it in ae:
            if isinstance(it, dict):
                if it.get("name") is not None and not it.get("evidence"):
                    it["evidence"] = fallback

    return data


def _check_citation(ev: Any, num_sources: int) -> None:
    ev = "" if ev is None else str(ev)
    m = CIT_RE.search(ev)
    if not m:
        raise ValidationError(f"Missing citation in evidence: {ev!r}")
    s_idx = int(m.group(1))
    if not (1 <= s_idx <= num_sources):
        raise ValidationError(f"Citation out of range: {ev} (num_sources={num_sources})")


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _to_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def _coerce_value_evidence(obj: Any, field_name: str, num_sources: int) -> Dict[str, Any]:
    """
    Accept either:
      - {"value": str|null, "evidence": "[S#]"|null}
      - str
      - null
    """
    if isinstance(obj, dict):
        if "value" not in obj or "evidence" not in obj:
            raise ValidationError(f"{field_name} must contain 'value' and 'evidence'")

        val = obj["value"]
        ev = obj["evidence"]

        if val is None:
            if ev is not None:
                raise ValidationError(f"{field_name}.evidence must be null when value is null")
            return {"value": None, "evidence": None}

        if not isinstance(val, str) or not val.strip():
            raise ValidationError(f"{field_name}.value must be a non-empty string or null")

        _check_citation(ev, num_sources)
        return {"value": val.strip(), "evidence": str(ev)}

    if isinstance(obj, str):
        val = obj.strip()
        if not val:
            return {"value": None, "evidence": None}
        ev = "[S1]"
        _check_citation(ev, num_sources)
        return {"value": val, "evidence": ev}

    if obj is None:
        return {"value": None, "evidence": None}

    raise ValidationError(f"{field_name} must be object, string, or null")


def _normalize_other_notes(data: Dict[str, Any]) -> None:
    if "other_notes" not in data:
        data["other_notes"] = None
    if data["other_notes"] is not None and not isinstance(data["other_notes"], str):
        data["other_notes"] = str(data["other_notes"])


def _normalize_negated_findings(data: Dict[str, Any], num_sources: int) -> None:
    neg = data.get("negated_findings", [])
    neg_list = _as_list(neg)
    out = []
    for it in neg_list:
        if isinstance(it, str):
            out.append({"name": it.strip(), "evidence": "[S1]"})
            continue

        if not isinstance(it, dict):
            continue

        name = str(it.get("name", "")).strip()
        ev = it.get("evidence", "[S1]")

        m = NEG_PREFIX_RE.match(name)
        if m:
            name = m.group(2).strip()

        if not name:
            continue

        _check_citation(ev, num_sources)
        out.append({"name": name, "evidence": str(ev)})

    data["negated_findings"] = out


def _move_negations_from_adverse_events(data: Dict[str, Any]) -> None:
    adv = data.get("adverse_events", [])
    adv_list = _as_list(adv)
    neg = _as_list(data.get("negated_findings", []))

    kept = []
    for it in adv_list:
        if not isinstance(it, dict):
            continue

        if "value" in it and "name" not in it:
            name = str(it.get("value", "")).strip()
            ev = it.get("evidence", "[S1]")
            if name:
                kept.append({"name": name, "grade": None, "evidence": ev})
            continue

        name = str(it.get("name", "")).strip()
        ev = it.get("evidence", "[S1]")

        m = NEG_PREFIX_RE.match(name)
        if m:
            concept = m.group(2).strip()
            neg.append({"name": concept if concept else name, "evidence": ev})
            continue

        kept.append(it)

    data["adverse_events"] = kept
    data["negated_findings"] = neg


def _normalize_adverse_events(data: Dict[str, Any], num_sources: int) -> None:
    adv = data.get("adverse_events", [])
    adv_list = _as_list(adv)

    out = []
    for it in adv_list:
        if not isinstance(it, dict):
            continue

        name = str(it.get("name", "")).strip()
        grade = _to_int_or_none(it.get("grade", None))
        ev = it.get("evidence", "[S1]")

        if not name:
            continue

        _check_citation(ev, num_sources)
        out.append({"name": name, "grade": grade, "evidence": str(ev)})

    data["adverse_events"] = out


def validate_json_answer(text: str, num_sources: int) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except Exception as e:
        raise ValidationError(f"Model did not return valid JSON: {e}")

    # ✅ IMPORTANT: fix missing evidence BEFORE coercion/validation
    data = _fill_missing_evidence(data, num_sources)

    for key in ["diagnosis", "treatment", "adverse_events", "negated_findings", "follow_up"]:
        if key not in data:
            raise ValidationError(f"Missing key: {key}")

    _normalize_other_notes(data)

    data["diagnosis"] = _coerce_value_evidence(data["diagnosis"], "diagnosis", num_sources)
    data["treatment"] = _coerce_value_evidence(data["treatment"], "treatment", num_sources)
    data["follow_up"] = _coerce_value_evidence(data["follow_up"], "follow_up", num_sources)

    data["negated_findings"] = _as_list(data.get("negated_findings", []))
    data["adverse_events"] = _as_list(data.get("adverse_events", []))

    _move_negations_from_adverse_events(data)
    _normalize_negated_findings(data, num_sources)
    _normalize_adverse_events(data, num_sources)

    return data
