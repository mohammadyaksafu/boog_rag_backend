"""
FAISS-backed vector store — one index per uploaded document, all kept on
disk permanently. Any number of documents can be "active" at once; their
indexes are kept loaded in memory together, and search() queries across
all of them, returning results tagged with which document they came from.
"""
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.pdf_processor import PageChunk

_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model, device=settings.device)
    return _embedder


def _index_path(doc_id: int) -> Path:
    return Path(settings.data_dir, "index", f"{doc_id}.faiss")


def _meta_path(doc_id: int) -> Path:
    return Path(settings.data_dir, "index", f"{doc_id}.meta.pkl")


class VectorStore:
    def __init__(self):
        # doc_id -> (faiss index, list[PageChunk])
        self._loaded: Dict[int, Tuple[faiss.Index, List[PageChunk]]] = {}

    def build(self, doc_id: int, chunks: List[PageChunk]):
        embedder = _get_embedder()
        texts = [c.text for c in chunks]
        embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        faiss.write_index(index, str(_index_path(doc_id)))
        with open(_meta_path(doc_id), "wb") as f:
            pickle.dump(chunks, f)

        self._loaded[doc_id] = (index, chunks)

    def load(self, doc_id: int) -> bool:
        if doc_id in self._loaded:
            return True
        if not _index_path(doc_id).exists() or not _meta_path(doc_id).exists():
            return False
        index = faiss.read_index(str(_index_path(doc_id)))
        with open(_meta_path(doc_id), "rb") as f:
            chunks = pickle.load(f)
        self._loaded[doc_id] = (index, chunks)
        return True

    def unload(self, doc_id: int):
        self._loaded.pop(doc_id, None)

    def set_active_set(self, doc_ids: List[int]):
        """Ensures exactly these doc_ids are loaded; unloads anything else."""
        for doc_id in list(self._loaded.keys()):
            if doc_id not in doc_ids:
                self.unload(doc_id)
        for doc_id in doc_ids:
            self.load(doc_id)

    def delete(self, doc_id: int):
        if _index_path(doc_id).exists():
            _index_path(doc_id).unlink()
        if _meta_path(doc_id).exists():
            _meta_path(doc_id).unlink()
        self.unload(doc_id)

    def is_ready(self) -> bool:
        return len(self._loaded) > 0

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, PageChunk, float]]:
        """Searches across ALL currently loaded (active) documents.
        Returns (doc_id, chunk, score), best matches first overall."""
        if not self.is_ready():
            return []
        embedder = _get_embedder()
        q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        all_results: List[Tuple[int, PageChunk, float]] = []
        for doc_id, (index, chunks) in self._loaded.items():
            k = min(top_k, len(chunks))
            if k == 0:
                continue
            scores, idxs = index.search(q_emb, k)
            for score, idx in zip(scores[0], idxs[0]):
                if idx == -1:
                    continue
                all_results.append((doc_id, chunks[idx], float(score)))

        all_results.sort(key=lambda r: r[2], reverse=True)
        return all_results[:top_k]


vector_store = VectorStore()