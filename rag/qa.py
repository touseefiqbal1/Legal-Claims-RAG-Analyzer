# rag/qa.py
from __future__ import annotations
from typing import Dict, List, Optional
from langchain_community.vectorstores import FAISS

from rag.extractors import extract_from_citations

def ask_with_citations(
    vs: FAISS,
    question: str,
    k: int = 5,
    source_filter: Optional[str] = None,  # <-- NEW
    fetch_k: int = 25,                    # <-- NEW: over-retrieve then filter
) -> Dict:
    """
    Retrieve chunks, optionally restricting to a single PDF (source_filter).
    We over-retrieve (fetch_k) then filter so we can still return k results.
    """
    results = vs.similarity_search_with_score(question, k=fetch_k)

    # Optional post-filter by source (filename)
    if source_filter:
        results = [(d, s) for (d, s) in results if d.metadata.get("source") == source_filter]

    # Trim back to k
    results = results[:k]

    citations: List[Dict] = []
    for rank, (doc, score) in enumerate(results, start=1):
        citations.append({
            "rank": rank,
            "source": doc.metadata.get("source", "unknown"),
            "path": doc.metadata.get("path"),
            "page": doc.metadata.get("page", "n/a"),
            "chunk_id": doc.metadata.get("chunk_id", "n/a"),
            "score": float(score),
            "snippet": doc.page_content[:350].replace("\n", " ").strip(),
            "text": doc.page_content,
        })

    extracted, hits = extract_from_citations(citations)

    # Build deterministic answer from extracted fields (keep your existing order/logic)
    if extracted:
        lines = []
        field_order = [
            ("claim_reference", "Claim reference"),
            ("policy_number", "Policy number"),
            ("incident_date", "Incident date"),
            ("incident_time", "Incident time"),
            ("incident_location", "Incident location"),
            ("police_reference", "Police reference"),
            ("total_claimed", "Total claimed"),
            ("repair_estimate", "Repair estimate"),
            ("hire_charges", "Hire charges"),
            ("general_damages", "General damages"),
            ("special_damages", "Special damages"),
            ("suggested_reserve", "Suggested reserve"),
            ("suggested_settlement", "Suggested settlement"),
            ("injuries", "Injuries"),
            ("fraud_indicators", "Fraud indicators"),
        ]
        for key, label in field_order:
            if key in extracted:
                lines.append(f"- **{label}:** {extracted[key]}")
        answer = "\n".join(lines)
    else:
        answer = "No extractable fields found in the retrieved evidence (try increasing top-k or asking a more specific question)."

    hit_map = [{
        "field": h.field,
        "value": h.value,
        "citation_rank": h.citation_rank,
        "page": h.page,
        "source": h.source,
        "snippet": h.snippet,
    } for h in hits]

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "extracted": extracted,
        "hit_map": hit_map,
    }
