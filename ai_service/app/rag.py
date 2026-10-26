"""RAG over user-uploaded coding guidelines (PDF / text / markdown) with Chroma."""
from __future__ import annotations

import hashlib
import io
import math
import re
import threading

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_settings


# ----------------------------------------------------------------- embeddings
class HashingEmbeddings(Embeddings):
    """Dependency-free bag-of-words embeddings (feature hashing). Good enough for keyword-ish
    guideline lookup, works offline, and is deterministic - used for tests and as an option."""

    dim = 384

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9_]{2,}", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class OnnxEmbeddings(Embeddings):
    """Chroma's bundled all-MiniLM-L6-v2 (ONNX). Downloads the model once on first use."""

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_store: Chroma | None = None
_lock = threading.Lock()


def get_store() -> Chroma:
    global _store
    with _lock:
        if _store is None:
            s = get_settings()
            emb: Embeddings = HashingEmbeddings() if s.embedding_backend == "hash" else OnnxEmbeddings()
            _store = Chroma(
                collection_name=f"guidelines_{s.embedding_backend}",
                embedding_function=emb,
                persist_directory=s.chroma_dir,
            )
        return _store


def reset_store() -> None:
    global _store
    _store = None


# ------------------------------------------------------------------ indexing
def extract_documents(filename: str, data: bytes) -> list[Document]:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"page": i}))
        return docs
    if name.endswith((".txt", ".md", ".markdown", ".rst")):
        return [Document(page_content=data.decode("utf-8", errors="ignore"), metadata={"page": 1})]
    raise ValueError("Unsupported file type. Upload a PDF, .txt or .md file.")


def index_guideline(guideline_id: str, user_id: int, title: str, filename: str, data: bytes) -> int:
    pages = extract_documents(filename, data)
    if not pages:
        raise ValueError("No readable text found in the file (scanned PDFs need OCR first).")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    ids = []
    for i, c in enumerate(chunks):
        c.metadata.update({"guideline_id": guideline_id, "user_id": str(user_id), "title": title, "source": filename})
        ids.append(f"{guideline_id}:{i}")
    store = get_store()
    for start in range(0, len(chunks), 64):
        store.add_documents(chunks[start:start + 64], ids=ids[start:start + 64])
    return len(chunks)


def delete_guideline(guideline_id: str) -> None:
    store = get_store()
    found = store.get(where={"guideline_id": guideline_id})
    if found.get("ids"):
        store.delete(ids=found["ids"])


# ----------------------------------------------------------------- retrieval
def has_guidelines(user_id: int) -> bool:
    return bool(get_store().get(where={"user_id": str(user_id)}, limit=1).get("ids"))


def get_user_guidelines(user_id: int) -> list[Document]:
    """Return every indexed passage belonging to this user, without relevance truncation."""
    result = get_store().get(
        where={"user_id": str(user_id)},
        include=["documents", "metadatas"],
    )
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(documents, metadatas)
        if text
    ]


def retrieve(user_id: int, query: str, k: int = 4) -> list[Document]:
    return get_store().similarity_search(query, k=k, filter={"user_id": str(user_id)})


def citation(doc: Document) -> str:
    title = doc.metadata.get("title") or doc.metadata.get("source", "guideline")
    return f"{title}, p.{doc.metadata.get('page', '?')}"
