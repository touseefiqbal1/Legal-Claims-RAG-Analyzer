# rag/extractors.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

MONEY_RE = re.compile(r"£\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")

@dataclass
class Hit:
    field: str
    value: str
    citation_rank: int
    page: Any
    source: str
    snippet: str

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _search_patterns(text: str, patterns: List[re.Pattern]) -> Optional[str]:
    for p in patterns:
        m = p.search(text)
        if m:
            # prefer named group "val" if present
            if "val" in m.groupdict():
                return _clean(m.group("val"))
            return _clean(m.group(0))
    return None

def _money_near_label(text: str, labels: List[str]) -> Optional[str]:
    """
    Find a £ amount near a label like 'Total Claimed' / 'Suggested Reserve'.
    """
    t = text
    for lab in labels:
        # label ... £amount (same line or nearby)
        p = re.compile(rf"{re.escape(lab)}.{0,80}({MONEY_RE.pattern})", re.IGNORECASE | re.DOTALL)
        m = p.search(t)
        if m:
            return _clean(m.group(1))
    return None

def extract_from_citations(citations: List[Dict]) -> Tuple[Dict[str, str], List[Hit]]:
    """
    Given citations (each with rank/source/page/snippet plus ideally evidence/fulltext),
    extract fields and return:
      - fields dict
      - list of Hit(field,value,citation metadata)
    """
    # Patterns geared to your synthetic generator format but robust-ish for real docs too.
    patterns = {
        "claim_reference": [
            re.compile(r"\bClaim Reference:\s*(?P<val>CLM-[A-Z]{3}-\d{6})\b", re.IGNORECASE),
            re.compile(r"\b(CL M|CLM)[-\s]?[A-Z]{3}[-\s]?\d{6}\b".replace(" ", ""), re.IGNORECASE),
        ],
        "policy_number": [
            re.compile(r"\bPolicy Number:\s*(?P<val>POL-\d{8})\b", re.IGNORECASE),
            re.compile(r"\bPOL-\d{8}\b", re.IGNORECASE),
        ],
        "police_reference": [
            re.compile(r"\bPolice Reference:\s*(?P<val>PNC/\d{4}/\d{7})\b", re.IGNORECASE),
            re.compile(r"\bPNC/\d{4}/\d{7}\b", re.IGNORECASE),
        ],
        "incident_date": [
            re.compile(r"\bIncident Date:\s*(?P<val>\d{4}-\d{2}-\d{2})\b", re.IGNORECASE),
        ],
        "incident_time": [
            re.compile(r"\bIncident Time:\s*(?P<val>\d{2}:\d{2})\b", re.IGNORECASE),
        ],
        "incident_location": [
            re.compile(r"\bLocation:\s*(?P<val>.+)", re.IGNORECASE),
        ],
    }

    money_labels = {
        "total_claimed": ["Total Claimed"],
        "suggested_reserve": ["Suggested Reserve", "Reserve"],
        "suggested_settlement": ["Suggested Settlement Range", "Settlement"],
        "repair_estimate": ["Repair Estimate"],
        "hire_charges": ["Total Hire Charges", "Hire Charges"],
        "general_damages": ["General Damages"],
        "special_damages": ["Special Damages"],
    }

    extracted: Dict[str, str] = {}
    hits: List[Hit] = []

    for c in citations:
        rank = int(c.get("rank", 0))
        src = c.get("source", "unknown")
        page = c.get("page", "n/a")

        # Prefer full chunk text if you included it; fallback to snippet.
        text = c.get("text") or c.get("chunk") or c.get("snippet") or ""
        text = text.strip()
        if not text:
            continue

        # simple fields
        for field, pats in patterns.items():
            if field in extracted:
                continue
            val = _search_patterns(text, pats)
            if val:
                # Location pattern grabs rest of line; trim common trailing junk
                if field == "incident_location":
                    val = val.split("Incident")[0].strip()
                extracted[field] = val
                hits.append(Hit(field, val, rank, page, src, _clean(text)[:220]))

        # money fields
        for field, labels in money_labels.items():
            if field in extracted:
                continue
            val = _money_near_label(text, labels)
            if val:
                extracted[field] = val
                hits.append(Hit(field, val, rank, page, src, _clean(text)[:220]))

        # list-ish fields (injuries / fraud indicators)
        # Injuries often appear as bullet lines; capture bullet-like lines in that section.
        if "injuries" not in extracted:
            if re.search(r"\bReported Injuries\b", text, re.IGNORECASE):
                bullets = re.findall(r"(?:^|\n)[•\-\*]\s*(.+)", text)
                if bullets:
                    vals = [ _clean(b) for b in bullets if len(_clean(b)) > 3 ]
                    if vals:
                        extracted["injuries"] = "; ".join(vals)
                        hits.append(Hit("injuries", extracted["injuries"], rank, page, src, _clean(text)[:220]))

        if "fraud_indicators" not in extracted:
            if re.search(r"\bFraud\b|\bindicators?\b|\btriage\b", text, re.IGNORECASE):
                bullets = re.findall(r"(?:^|\n)[•\-\*]\s*(.+)", text)
                if bullets:
                    vals = [ _clean(b) for b in bullets if len(_clean(b)) > 3 ]
                    # keep only if it looks like fraud/red-flag content
                    vals = [v for v in vals if re.search(r"claim|witness|hire|damage|notification|inconsistent|prior", v, re.IGNORECASE)]
                    if vals:
                        extracted["fraud_indicators"] = "; ".join(vals[:6])
                        hits.append(Hit("fraud_indicators", extracted["fraud_indicators"], rank, page, src, _clean(text)[:220]))

    return extracted, hits
