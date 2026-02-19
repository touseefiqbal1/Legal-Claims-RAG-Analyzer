from __future__ import annotations
from typing import List, Tuple
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    # Fast, local, good enough for demo
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_faiss_index(chunks: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)

def save_index(vs: FAISS, index_dir: str) -> None:
    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)

def load_index(index_dir: str) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
