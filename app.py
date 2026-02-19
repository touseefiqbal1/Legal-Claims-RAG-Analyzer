# app.py
import os
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import fitz  # PyMuPDF
import pandas as pd

from rag.ingest_pdf import load_pdf_as_documents
from rag.chunking import chunk_documents
from rag.index_faiss import build_faiss_index, save_index, load_index
from rag.qa import ask_with_citations

# -----------------------------
# Config (ABSOLUTE PATHS)
# -----------------------------
st.set_page_config(page_title="Court Pack RAG Assistant", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / "data" / "sample_pdfs").resolve()
INDEX_DIR = (BASE_DIR / "indexes" / "faiss_index").resolve()
INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_MANIFEST = (BASE_DIR / "indexes" / "manifest.json").resolve()

# -----------------------------
# PDF Viewer (robust; no cached doc handles)
# -----------------------------
def get_pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()

@st.cache_data(show_spinner=False)
def render_pdf_page_to_png_bytes(pdf_path: str, page_num_1based: int, zoom: float, mtime: float) -> bytes:
    """
    Render a given PDF page to PNG bytes.
    mtime is included so cache invalidates if the PDF file changes.
    """
    doc = fitz.open(pdf_path)
    try:
        page_index = page_num_1based - 1
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(f"Page out of range: {page_num_1based} (PDF has {len(doc)} pages)")
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()

def resolve_pdf_path_from_citation(c: dict) -> str | None:
    """
    Prefer citation['path'] if present, else try:
      - citation['source'] as path (relative to BASE_DIR)
      - basename under DATA_DIR
      - recursive search under DATA_DIR
    Always returns an ABSOLUTE path if found.
    """
    p = c.get("path")
    if p:
        p_abs = str(Path(p).expanduser().resolve())
        if os.path.exists(p_abs):
            return p_abs

    src = c.get("source")
    if not src:
        return None

    src_path = Path(src).expanduser()
    if not src_path.is_absolute():
        src_path = (BASE_DIR / src_path).resolve()
    else:
        src_path = src_path.resolve()

    if src_path.exists():
        return str(src_path)

    candidate = (DATA_DIR / Path(src).name).resolve()
    if candidate.exists():
        return str(candidate)

    matches = list(DATA_DIR.rglob(Path(src).name))
    if matches:
        return str(matches[0].resolve())

    return None

# -----------------------------
# Index caching
# -----------------------------
@st.cache_resource(show_spinner=False)
def cached_load_index(index_dir: str):
    return load_index(index_dir)

# -----------------------------
# Session state
# -----------------------------
if "viewer_pdf_path" not in st.session_state:
    st.session_state.viewer_pdf_path = None
if "viewer_page" not in st.session_state:
    st.session_state.viewer_page = None
if "viewer_zoom" not in st.session_state:
    st.session_state.viewer_zoom = 2.0

# -----------------------------
# Viewer setter (used by on_click)
# -----------------------------
def set_viewer(pdf_path: str, page: int):
    st.session_state.viewer_pdf_path = pdf_path
    st.session_state.viewer_page = page

# -----------------------------
# Evaluation helpers (manifest path resolution + hit@k)
# -----------------------------
def resolve_manifest_path(manifest_file: str, p: str, fallback_dir: Optional[str] = None) -> str:
    """
    Resolve a path referenced in manifest.json.

    Rules:
      - If p is absolute and exists -> use it
      - Else resolve relative to the manifest file's directory
      - If still missing and fallback_dir provided -> try fallback_dir / basename(p)
    Returns the best candidate path (absolute string).
    """
    pth = Path(p).expanduser()
    if pth.is_absolute():
        p_abs = pth.resolve()
        if p_abs.exists():
            return str(p_abs)

    manifest_dir = Path(manifest_file).resolve().parent
    candidate = (manifest_dir / pth).resolve()
    if candidate.exists():
        return str(candidate)

    if fallback_dir:
        fb = (Path(fallback_dir).resolve() / pth.name).resolve()
        if fb.exists():
            return str(fb)

    return str(candidate)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def _contains_any(citations: List[Dict], target: str) -> bool:
    t = _norm(target)
    if not t:
        return False
    for c in citations:
        txt = (c.get("text") or c.get("snippet") or "")
        if t in _norm(txt):
            return True
    return False

def _money_variants(x: float) -> List[str]:
    return [f"£{x:,.2f}", f"£{x:.2f}", f"{x:,.2f}", f"{x:.2f}"]

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

def _expected_strings(gt: Dict, field: str) -> List[str]:
    v = gt.get(field)
    if v is None:
        return []
    if field in ("total_claimed", "reserve_recommendation"):
        try:
            x = float(v)
            return _money_variants(x)
        except Exception:
            return [str(v)]
    return [str(v)]

@st.cache_data(show_spinner=False)
def run_evaluation_cached(
    index_dir: str,
    manifest_path: str,
    k: int,
    fetch_k: int,
    restrict_to_pack: bool,
    index_marker: float,
    manifest_marker: float,
    fallback_dir: str,
) -> Dict:
    """
    Cached evaluation run. Markers ensure invalidation when index or manifest changes.
    """
    vs = cached_load_index(index_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    per_field_hits = {field: 0 for field, _ in EVAL_FIELDS}
    per_field_total = {field: 0 for field, _ in EVAL_FIELDS}
    pack_rows = []

    for pack in manifest:
        gt_path = resolve_manifest_path(manifest_path, pack["ground_truth"], fallback_dir=fallback_dir)
        pdf_path = resolve_manifest_path(manifest_path, pack["pdf"], fallback_dir=fallback_dir)

        gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
        pdf_name = os.path.basename(pdf_path)
        source_filter = pdf_name if restrict_to_pack else None

        pack_hit = 0
        pack_total = 0

        for field, question in EVAL_FIELDS:
            exp_list = _expected_strings(gt, field)
            if not exp_list:
                continue

            out = ask_with_citations(vs, question, k=k, source_filter=source_filter, fetch_k=fetch_k)
            hit = any(_contains_any(out["citations"], exp) for exp in exp_list)

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
    overall_rate = (overall_hits / overall_total) if overall_total else None

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
        "fallback_dir": str(Path(fallback_dir).resolve()),
        "overall": {"hits": overall_hits, "total": overall_total, "hit_rate": overall_rate},
        "per_field": per_field,
        "per_pack": pack_rows,
    }

# -----------------------------
# UI
# -----------------------------
st.title("Court Pack RAG Assistant (FAISS + Pack Selector + Page Viewer + Rule-Based Extraction)")

# Sidebar: data + index build
st.sidebar.header("Data")
st.sidebar.write(f"PDF folder: `{DATA_DIR}`")

pdfs = sorted(glob.glob(str(DATA_DIR / "*.pdf")))
st.sidebar.write(f"Found **{len(pdfs)}** PDFs")

st.sidebar.divider()
st.sidebar.subheader("Index")

if st.sidebar.button("Build / Rebuild FAISS index", use_container_width=True):
    if not pdfs:
        st.sidebar.error(f"No PDFs found in {DATA_DIR}")
    else:
        with st.spinner("Loading PDFs (page-level)..."):
            all_pages = []
            for p in pdfs:
                all_pages.extend(load_pdf_as_documents(str(Path(p).resolve())))

        with st.spinner("Chunking..."):
            chunks = chunk_documents(all_pages, chunk_size=1000, chunk_overlap=150)

        with st.spinner("Building FAISS index..."):
            vs_tmp = build_faiss_index(chunks)

        with st.spinner("Saving index..."):
            save_index(vs_tmp, str(INDEX_DIR))

        cached_load_index.clear()
        st.sidebar.success(f"Index built ✅ Pages: {len(all_pages)} | Chunks: {len(chunks)} | Saved: {INDEX_DIR}")

# Sidebar: Evaluation (NEW)
st.sidebar.divider()
st.sidebar.subheader("Evaluation")

manifest_path = st.sidebar.text_input(
    "Manifest path",
    value=str(DEFAULT_MANIFEST),
    help="manifest.json with pdf + ground_truth references (relative paths are resolved relative to manifest folder, with fallback to /indexes).",
)

eval_k = st.sidebar.slider("Eval: top-k", 3, 10, 5)
eval_fetch_k = st.sidebar.slider("Eval: fetch_k", 20, 100, 50, help="Over-retrieve, then filter down to k.")
restrict_to_pack = st.sidebar.checkbox("Restrict retrieval to pack", value=True)

run_eval = st.sidebar.button("Run evaluation", use_container_width=True)

# Main layout
left, right = st.columns([1, 1], gap="large")

with right:
    st.subheader("Ask a Question")

    # Pack selector
    pack_options = ["All packs"] + [Path(p).name for p in pdfs]
    selected_pack = st.selectbox("Case pack", pack_options, index=0)
    source_filter = None if selected_pack == "All packs" else selected_pack
    st.caption(f"Retrieval scope: **{selected_pack}**")

    q = st.text_input("Question", placeholder="e.g., What is the total claimed amount?")
    k = st.slider("Top-k retrieved chunks", 3, 10, 5)
    zoom = st.slider("Viewer zoom", 1.0, 3.0, float(st.session_state.viewer_zoom), 0.25)

    ask_clicked = st.button("Ask", type="primary", use_container_width=True)

    # Evaluation output area
    if run_eval:
        if not INDEX_DIR.exists():
            st.error("Index not found. Build the index first.")
        elif not Path(manifest_path).exists():
            st.error(f"Manifest not found: {manifest_path}")
        else:
            with st.spinner("Running evaluation..."):
                index_marker = (INDEX_DIR / "index.faiss").stat().st_mtime if (INDEX_DIR / "index.faiss").exists() else INDEX_DIR.stat().st_mtime
                manifest_marker = Path(manifest_path).stat().st_mtime
                fallback_dir = str((BASE_DIR / "indexes").resolve())

                report = run_evaluation_cached(
                    index_dir=str(INDEX_DIR),
                    manifest_path=str(Path(manifest_path).resolve()),
                    k=int(eval_k),
                    fetch_k=int(eval_fetch_k),
                    restrict_to_pack=bool(restrict_to_pack),
                    index_marker=float(index_marker),
                    manifest_marker=float(manifest_marker),
                    fallback_dir=fallback_dir,
                )

            st.markdown("## Evaluation results")
            overall = report["overall"]
            st.metric("Hit rate", f"{(overall['hit_rate']*100):.1f}%" if overall["hit_rate"] is not None else "n/a")
            st.caption(
                f"hits={overall['hits']} / total={overall['total']} | "
                f"k={report['k']} | fetch_k={report['fetch_k']} | restrict_to_pack={report['restrict_to_pack']}"
            )

            df_field = pd.DataFrame(report["per_field"]).sort_values("hit_rate", ascending=True)
            df_pack = pd.DataFrame(report["per_pack"]).sort_values("hit_rate", ascending=True)

            st.markdown("### Per-field hit rates")
            st.dataframe(df_field, use_container_width=True)

            st.markdown("### Per-pack hit rates")
            st.dataframe(df_pack, use_container_width=True)

            out_path = (BASE_DIR / "evaluation_report.json").resolve()
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            st.caption(f"Saved report: `{out_path}`")

            st.divider()

    if ask_clicked and q.strip():
        st.session_state.viewer_zoom = float(zoom)

        if not INDEX_DIR.exists():
            st.error("Index not found. Build the index first (sidebar).")
        else:
            vs = cached_load_index(str(INDEX_DIR))
            with st.spinner("Retrieving + extracting..."):
                out = ask_with_citations(vs, q, k=k, source_filter=source_filter, fetch_k=50)

            st.markdown("### Answer (rule-based)")
            st.markdown(out.get("answer", ""))

            st.markdown("### Extracted fields (with supporting citation)")
            hit_map = out.get("hit_map", []) or []
            if hit_map:
                for h in hit_map:
                    st.write(
                        f"**{h['field']}** = {h['value']}  → "
                        f"citation **[{h['citation_rank']}]** ({h['source']}, p.{h['page']})"
                    )
            else:
                st.caption("No fields extracted. Try a more specific question or increase Top-k.")

            st.markdown("### Citations (click to view page)")
            citations = out.get("citations", []) or []

            if source_filter and not citations:
                st.warning("No citations returned for this pack. Try increasing Top-k or ask a more specific question.")

            for c in citations:
                pdf_path = resolve_pdf_path_from_citation(c)
                page = c.get("page")

                header = (
                    f"[{c.get('rank','?')}] {c.get('source','')} | p.{page} | "
                    f"chunk {c.get('chunk_id','?')} | score {c.get('score',0):.4f}"
                )
                st.markdown(f"**{header}**")
                st.caption(c.get("snippet", ""))

                page_ok = isinstance(page, int) or (isinstance(page, str) and str(page).isdigit())
                page_int = int(page) if page_ok else None
                can_view = bool(pdf_path) and page_ok

                view_key = f"view::{c.get('rank')}::{c.get('source')}::{c.get('page')}::{c.get('chunk_id')}"

                st.button(
                    "View page",
                    key=view_key,
                    disabled=not can_view,
                    on_click=set_viewer,
                    args=(str(pdf_path), int(page_int)) if can_view else None,
                )

                if not pdf_path:
                    st.caption("⚠️ Could not resolve PDF path for this citation. (Check ingestion `metadata['path']` and rebuild index.)")

                st.divider()

            with st.expander("Evidence (full retrieved chunks)"):
                for c in citations:
                    st.write(f"**Citation [{c.get('rank','?')}]** — {c.get('source','')} p.{c.get('page','?')}")
                    st.write(c.get("text", ""))
                    st.divider()

with left:
    st.subheader("Page Viewer")

    if st.button("Clear viewer", use_container_width=True):
        st.session_state.viewer_pdf_path = None
        st.session_state.viewer_page = None

    pdf_path = st.session_state.viewer_pdf_path
    page = st.session_state.viewer_page
    zoom_state = float(st.session_state.viewer_zoom)

    if pdf_path and page:
        if not os.path.exists(pdf_path):
            st.error(f"Viewer path does not exist: {pdf_path}")
        else:
            try:
                page_count = get_pdf_page_count(pdf_path)
                st.caption(f"{Path(pdf_path).name} — page {page} of {page_count}")

                new_page = st.number_input("Go to page", 1, page_count, value=int(page), step=1)
                if int(new_page) != int(page):
                    st.session_state.viewer_page = int(new_page)

                mtime = os.path.getmtime(pdf_path)
                img = render_pdf_page_to_png_bytes(pdf_path, int(st.session_state.viewer_page), zoom_state, mtime)
                st.image(img, use_container_width=True)

            except Exception as e:
                st.error(f"Could not render page: {e}")
    else:
        st.info("Click “View page” on a citation to preview the cited PDF page here.")
