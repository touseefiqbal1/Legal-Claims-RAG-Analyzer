from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked: List[Document] = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, ch in enumerate(chunks):
            md = dict(d.metadata)
            md["chunk_id"] = i
            chunked.append(Document(page_content=ch, metadata=md))
    return chunked
