from __future__ import annotations
import re
from typing import List, Dict, Optional

STOPWORDS = {"is", "are", "was", "were", "present", "evidence", "of", "the", "a", "an", "in", "no"}

def extract_focus_term(question: str) -> Optional[str]:
    """
    Very lightweight heuristic:
    If question matches 'Is X present?' or 'Is there ... X ...?'
    return X as focus term.
    """
    q = question.strip().lower()

    m = re.search(r"is\s+([a-z0-9\- ]+?)\s+present\??$", q)
    if m:
        term = m.group(1).strip()
        if term and term not in STOPWORDS:
            return term

    # also handle: "Is there evidence of X?"
    m = re.search(r"evidence of\s+([a-z0-9\- ]+?)\??$", q)
    if m:
        term = m.group(1).strip()
        if term and term not in STOPWORDS:
            return term

    return None


def sources_contain_term(hits: List[Dict], term: str) -> bool:
    term = term.lower().strip()
    if not term:
        return True
    for h in hits:
        if term in (h.get("text", "").lower()):
            return True
    return False
