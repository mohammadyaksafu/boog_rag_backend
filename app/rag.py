from typing import List

import google.generativeai as genai
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Document
from app.models import SourceChunk, PageRef
from app.vector_store import vector_store

_gemini_configured = False


def _ensure_gemini():
    global _gemini_configured
    if not _gemini_configured:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment (.env)")
        genai.configure(api_key=settings.gemini_api_key)
        _gemini_configured = True


SYSTEM_INSTRUCTION = """You are a precise document Q&A assistant.
You must answer ONLY using the provided excerpts from the active book(s).
Rules:
- If the answer is present in the excerpts, answer concisely and accurately, staying faithful to the text.
- Always mention which book and page(s) support your answer using the [Book: ... | Page: N] tags given in the excerpts.
- If multiple active books are relevant, synthesize across them and be clear which book each fact came from.
- If the excerpts do not contain enough information to answer, say so plainly instead of guessing.
- Do not use outside/general knowledge beyond what is in the excerpts.
"""


def answer_question(question: str, db: Session, top_k: int = 5) -> dict:
    if not vector_store.is_ready():
        return {
            "answer": "No active book selected. Please upload and activate at least one book first.",
            "pages": [],
            "sources": [],
        }

    results = vector_store.search(question, top_k=top_k)
    if not results:
        return {
            "answer": "I couldn't find relevant content in the active book(s).",
            "pages": [],
            "sources": [],
        }

    doc_ids = {doc_id for doc_id, _, _ in results}
    doc_names = {
        d.id: d.filename
        for d in db.query(Document).filter(Document.id.in_(doc_ids)).all()
    }

    context_blocks = []
    sources: List[SourceChunk] = []
    pages_seen: List[PageRef] = []
    seen_keys = set()

    for doc_id, chunk, score in results:
        filename = doc_names.get(doc_id, f"Document {doc_id}")
        context_blocks.append(f"[Book: {filename} | Page: {chunk.page}]\n{chunk.text}")
        sources.append(SourceChunk(
            doc_id=doc_id, filename=filename, page=chunk.page,
            text=chunk.text[:400], score=round(score, 4),
        ))
        key = (doc_id, chunk.page)
        if key not in seen_keys:
            seen_keys.add(key)
            pages_seen.append(PageRef(doc_id=doc_id, filename=filename, page=chunk.page))

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""{SYSTEM_INSTRUCTION}

Excerpts from the active book(s):
{context}

Question: {question}

Answer (cite like (Book Title, p. X)):"""

    _ensure_gemini()
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(prompt)
    answer_text = response.text if hasattr(response, "text") else str(response)

    return {
        "answer": answer_text.strip(),
        "pages": pages_seen,
        "sources": sources,
    }