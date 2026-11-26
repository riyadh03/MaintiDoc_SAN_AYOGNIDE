"""Extraction DÉTERMINISTE des paramètres d'outils (équipement, type d'action, codes).

Pourquoi pas le LLM : sur CPU, un appel Qwen coûte 10–20 s, et un modèle 3B confond
facilement « contrôleur B » et « CTRL-A ». Ces décisions engagent la GMAO : elles sont
prises par du code testable. Le LLM a déjà fait sa part (l'intention) au routage.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

# Un identifiant du manuel (AL-01, DIAG-FC-01, INC-104, FIG-02, DISC-02…)
MANUAL_CODE = re.compile(r"\b(?:AL|DIAG|INC|FIG|DISC|CAS)-[A-Z]*-?\d+\b", re.IGNORECASE)

# Du plus précis au plus général : le premier motif trouvé désigne l'équipement
_EQUIPMENT_PATTERNS = [
    (re.compile(r"\bcontr[ôo]leur\s+([AB])\b", re.IGNORECASE), lambda m: f"CTRL-{m.group(1).upper()}"),
    (re.compile(r"\bCTRL-([AB])\b", re.IGNORECASE), lambda m: f"CTRL-{m.group(1).upper()}"),
    (re.compile(r"\bfabric\s+(?:FC\s+)?([AB])\b", re.IGNORECASE), lambda m: f"FABRIC-{m.group(1).upper()}"),
    (re.compile(r"\bdisque\s+([A-Z]\d{2})\b", re.IGNORECASE), lambda m: f"DISK-{m.group(1).upper()}"),
    (re.compile(r"\bDISK-([A-Z]\d{2})\b", re.IGNORECASE), lambda m: f"DISK-{m.group(1).upper()}"),
    (re.compile(r"\bVOL-[A-Z0-9]+\b", re.IGNORECASE), lambda m: m.group(0).upper()),
    (re.compile(r"\bSAN-LAB[- ]?2000\b|\bbaie\b", re.IGNORECASE), lambda m: "SAN-LAB-2000"),
]

_TICKET_REQUEST = re.compile(r"\bticket|\bintervention|\bdemande d'intervention|\bsignal(?:er|ement)\b", re.IGNORECASE)
# Manuel §5 : « Ne pas exécuter de commande destructive, de failover ou de reconfiguration »
_FORBIDDEN_ACTION = re.compile(
    r"failover|bascul|zoning|supprim|effac|formater|reconfigur|red[ée]marr|reboot|"
    r"lun\s*masking|d[ée]sactiv|arr[êe]te[rz]?\b",
    re.IGNORECASE,
)
_HISTORY = re.compile(r"historique|ant[ée]c[ée]dent|pass[ée]|intervention[s]? (?:pr[ée]c[ée]dente|ant[ée]rieure)", re.IGNORECASE)

DEFAULT_EQUIPMENT = "SAN-LAB-2000"
DEFAULT_PRIORITY = "P3"


def extract_equipment(text: str) -> Optional[str]:
    for pattern, to_id in _EQUIPMENT_PATTERNS:
        m = pattern.search(text)
        if m:
            return to_id(m)
    return None


def extract_codes(text: str) -> List[str]:
    seen: List[str] = []
    for m in MANUAL_CODE.finditer(text):
        code = m.group(0).upper()
        if code not in seen:
            seen.append(code)
    return seen


def classify_action(text: str) -> str:
    """``ticket`` (création soumise à approbation), ``forbidden`` (interdit par le manuel §5)
    ou ``unsupported`` (autre action, non prise en charge).

    Une demande de TICKET passe en premier : « crée un ticket pour un failover » documente
    une intervention qu'un technicien habilité réalisera ; MaintiDoc n'exécute rien.
    """
    if _TICKET_REQUEST.search(text):
        return "ticket"
    if _FORBIDDEN_ACTION.search(text):
        return "forbidden"
    return "unsupported"


def wants_history(text: str) -> bool:
    return bool(_HISTORY.search(text))


def draft_title(equipment_label: str, codes: List[str]) -> str:
    alarms = [c for c in codes if c.startswith("AL-")]
    if alarms:
        return f"Alerte {alarms[0]} sur {equipment_label}"  # format du ticket exemple du manuel (p.7)
    return f"Intervention {equipment_label}"


def idempotency_key(thread_id: str, draft: Dict[str, Any]) -> str:
    """Même conversation + même brouillon approuvé ⇒ même clé ⇒ jamais de doublon."""
    fields = {k: draft.get(k) for k in ("equipment_id", "title", "description", "priority", "evidence")}
    raw = json.dumps({"thread_id": thread_id, **fields}, sort_keys=True, ensure_ascii=False)
    return "mdoc-" + hashlib.sha256(raw.encode()).hexdigest()[:32]
