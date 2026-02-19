# evaluate.py
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rag.index_faiss import load_index
from rag.qa import ask_with_citations

# -----------------------------
# Helpers
# -----------------------------
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def contains_any(citations: List[Dict], target: str) -> bool:
    t = norm(target)
    if not t:
        return False
    for c in citations:
        txt = (c.get("text") or c.get("snippet") or "")
        if t in norm(txt):
            return True
    return False

def money_variants(x: float) -> List[str]:
    return [
        f"£{x:,.2f}",
        f"£{x:.2f}",
        f"{x:,.2f}",
        f"{x:.2f}",
    ]

def resolve_manifest_path(manifest_file: str, p: str, fallback_dir: Optional[str] = None) -> str:
    """
    Resolve a path referenced in manifest.json.

    Rules:
      - If p is absolute and exists -> use it
      - Else resolve relative to the manifest file's directory
      - If still missing and fallback_dir provided -> try fallback_dir / basename(p)
    Returns the best candidate path (absolute).
    """
    pth = Path(p).expanduser()

    if pth.is_absolute():
        p_abs = pth.resolve()
        if p_abs.exists():
            return str(p_abs)
    # Relative: resolve against manifest directory
    manifest_dir = Path(manifest_file).resolve().parent
    candidate = (manifest_dir / pth).resolve()
    if candidate.exists():
        return str(candidate)

    if fallback_dir:
        fb = (Path(fallback_dir).resolve() / pth.name).resolve()
        if fb.exists():
            return str(fb)

    # Return the manifest-relative candidate for easier debugging
    return str(candidate)

# -----------------------------
# Evaluation spec
# -----------------------------
EVAL_FIELDS: List[Tuple[str, str]] = [
    ("claim_reference", "What is the claim reference?"),
    ("policy_number", "What is the policy number?"),
    ("incident_date", "On what date did the incident occur?"),
    ("incident_time", "What time did the incident occur?"),
    ("incident_location", "Where did the incident occur?"),
    ("police_reference", "What is the police reference?"),
    ("total_claimed", "What is the total claimed amount?"),
    ("reserve_recommendation", "What is the suggested reserve?"),
]

def expected_strings(gt: Dict, field: str) -> List[str]:
    v = gt.get(field)
    if v is None:
        return []
    if field in ("total_claimed", "reserve_recommendation"):
        try:
            x = float(v)
            return money_variants(x)
        except Exception:
            return [str(v)]
    return [str(v)]

# -----------------------------
# Main evaluation
# -----------------------------
def evaluate(
    index_dir: str,
    manifest_path: str,
    k: int = 5,
    fetch_k: int = 50,
    restrict_to_pack: bool = True,
    fallback_dir: Optional[str] = None,
) -> Dict:
    """
    Computes Hit@k over a set of packs defined in manifest.json.

    restrict_to_pack=True:
      - filters retrieval to the pack's filename (metadata['source']) to avoid cross-pack contamination
    fetch_k:
      - over-retrieve and then filter to ensure we still return k items after source filtering
    fallback_dir:
      - optional directory to try if manifest paths are stale (e.g., files moved to /indexes)
    """
    vs = load_index(index_dir)

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    per_field_hits = {field: 0 for field, _ in EVAL_FIELDS}
    per_field_total = {field: 0 for field, _ in EVAL_FIELDS}
    pack_rows = []

    for pack in manifest:
        gt_path = resolve_manifest_path(manifest_path, pack["ground_truth"], fallback_dir=fallback_dir)
        pdf_path = resolve_manifest_path(manifest_path, pack["pdf"], fallback_dir=fallback_dir)

        gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
        pdf_name = os.path.basename(pdf_path)

        pack_hit = 0
        pack_total = 0

        for field, question in EVAL_FIELDS:
            exp_list = expected_strings(gt, field)
            if not exp_list:
                continue

            source_filter = pdf_name if restrict_to_pack else None
            out = ask_with_citations(vs, question, k=k, source_filter=source_filter, fetch_k=fetch_k)

            hit = any(contains_any(out["citations"], exp) for exp in exp_list)

            per_field_total[field] += 1
            per_field_hits[field] += int(hit)

            pack_total += 1
            pack_hit += int(hit)

        pack_rows.append({
            "claim_reference": gt.get("claim_reference"),
            "pdf": pdf_name,
            "hits": pack_hit,
            "total": pack_total,
            "hit_rate": (pack_hit / pack_total) if pack_total else None,
        })

    overall_hits = sum(per_field_hits.values())
    overall_total = sum(per_field_total.values())
    overall_rate = overall_hits / overall_total if overall_total else None

    per_field = []
    for f, _ in EVAL_FIELDS:
        tot = per_field_total[f]
        hit = per_field_hits[f]
        per_field.append({
            "field": f,
            "hits": hit,
            "total": tot,
            "hit_rate": (hit / tot) if tot else None,
        })

    return {
        "k": k,
        "fetch_k": fetch_k,
        "restrict_to_pack": restrict_to_pack,
        "manifest_path": str(Path(manifest_path).resolve()),
        "fallback_dir": str(Path(fallback_dir).resolve()) if fallback_dir else None,
        "overall": {"hits": overall_hits, "total": overall_total, "hit_rate": overall_rate},
        "per_field": per_field,
        "per_pack": pack_rows,
    }

# -----------------------------
# CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=str,
        default=str((BASE_DIR / "indexes" / "manifest.json").resolve()),
        help="Path to manifest.json (PDF + ground-truth references).",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=str((BASE_DIR / "indexes" / "faiss_index").resolve()),
        help="Directory containing saved FAISS index.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fetch_k", type=int, default=50)
    parser.add_argument("--restrict_to_pack", action="store_true", default=True)
    parser.add_argument(
        "--fallback_dir",
        type=str,
        default=str((BASE_DIR / "indexes").resolve()),
        help="Fallback directory if manifest paths are stale (e.g., files moved).",
    )
    args = parser.parse_args()

    report = evaluate(
        index_dir=str(Path(args.index_dir).resolve()),
        manifest_path=str(Path(args.manifest).resolve()),
        k=int(args.k),
        fetch_k=int(args.fetch_k),
        restrict_to_pack=bool(args.restrict_to_pack),
        fallback_dir=str(Path(args.fallback_dir).resolve()) if args.fallback_dir else None,
    )

    out_path = (BASE_DIR / "evaluation_report.json").resolve()
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print("Overall hit_rate@k:", report["overall"]["hit_rate"])
    print("Per-field hit rates:")
    for row in report["per_field"]:
        print(f"  {row['field']}: {row['hit_rate']}")
