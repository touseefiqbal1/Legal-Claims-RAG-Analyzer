from __future__ import annotations
from typing import List
import os
import fitz  # PyMuPDF
from langchain_core.documents import Document

def load_pdf_as_documents(pdf_path: str) -> List[Document]:
    docs: List[Document] = []
    pdf = fitz.open(pdf_path)

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        text = (page.get_text("text") or "").strip()
        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(pdf_path),  # nicer display
                    "path": pdf_path,                      # <-- REQUIRED for viewer
                    "page": page_index + 1,
                },
            )
        )

    pdf.close()
    return docs
