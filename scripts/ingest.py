"""
Run this LOCALLY before deploying (optional — only needed if you want to
pre-process a book without going through the web UI):

    python scripts/ingest.py path/to/book.pdf

It processes the PDF (text + local VLM fallback for image pages), builds
the FAISS index, and inserts a row into the SQLite DB — the same result
as uploading through /documents/upload, without needing the server
running or an auth token. The new book is marked active.
"""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.database import init_db, SessionLocal, Document
from app.pdf_processor import extract_pdf_structure, chunk_pages
from app.vector_store import vector_store


def main(pdf_path: str):
    init_db()
    db = SessionLocal()

    pdf_dir = Path(settings.data_dir, "pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(pdf_path).name
    stored_path = pdf_dir / filename
    counter = 1
    stem, suffix = stored_path.stem, stored_path.suffix
    while stored_path.exists():
        stored_path = pdf_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy(pdf_path, stored_path)

    print("Extracting structure (this may download the local VLM once)...")
    pages = extract_pdf_structure(str(stored_path))
    chunks = chunk_pages(pages)

    doc = Document(
        filename=filename,
        stored_path=str(stored_path),
        page_count=len(pages),
        status="ready",
        uploaded_by=None,
        is_active=True,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # only one active book at a time
    db.query(Document).filter(Document.id != doc.id).update({Document.is_active: False})
    db.commit()

    print(f"Building FAISS index from {len(chunks)} chunks...")
    vector_store.build(doc.id, chunks)
    print(f"Done. '{filename}' ({len(pages)} pages) is now active. Total books in DB: {db.query(Document).count()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/ingest.py path/to/book.pdf")
        sys.exit(1)
    main(sys.argv[1])