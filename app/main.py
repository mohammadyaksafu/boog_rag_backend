import shutil
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.config import settings
from app.database import init_db, get_db, User, Document, ChatMessage, SessionLocal
from app.auth import (
    authenticate_user, create_access_token, get_current_user,
    hash_password, bootstrap_admin,get_current_user_from_query_token,
)
from app.models import (
    Token, UserCreate, UserOut, DocumentOut, QueryRequest, QueryResponse, ChatHistoryItem,
)
from app.pdf_processor import extract_pdf_structure, chunk_pages
from app.vector_store import vector_store
from app.rag import answer_question

app = FastAPI(title="Book RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.frontend_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    db = SessionLocal()
    try:
        bootstrap_admin(db)
    finally:
        db.close()


# ---------------------- Auth ----------------------

@app.post("/auth/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(username=payload.username, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user.username})
    return Token(access_token=token)


@app.get("/auth/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user


# ---------------------- Query ----------------------

@app.post("/query", response_model=QueryResponse)
def query_document(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    active_docs = db.query(Document).filter(Document.is_active == True).all()  # noqa: E712
    if not active_docs:
        raise HTTPException(status_code=400, detail="No active book(s). Upload and activate at least one book first.")

    vector_store.set_active_set([d.id for d in active_docs])
    result = answer_question(payload.question, db=db, top_k=payload.top_k)

    pages_str = ",".join(f"{p.doc_id}:{p.page}" for p in result["pages"])
    primary_doc_id = active_docs[0].id
    db.add(ChatMessage(user_id=current_user.id, document_id=primary_doc_id, role="user", content=payload.question))
    db.add(ChatMessage(
        user_id=current_user.id, document_id=primary_doc_id, role="assistant",
        content=result["answer"], pages=pages_str,
    ))
    db.commit()

    return QueryResponse(**result)


@app.get("/chat/history", response_model=list[ChatHistoryItem])
def chat_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    active_ids = [d.id for d in db.query(Document).filter(Document.is_active == True).all()]  # noqa: E712
    if not active_ids:
        return []
    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == current_user.id, ChatMessage.document_id.in_(active_ids))
        .order_by(ChatMessage.id.asc())
        .all()
    )
    return msgs

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------- Documents (multi-book) ----------------------

@app.post("/documents/upload", response_model=DocumentOut)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not settings.enable_upload_delete:
        raise HTTPException(status_code=403, detail="Uploads are disabled on this deployment")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_dir = Path(settings.data_dir, "pdfs")
    stored_path = pdf_dir / file.filename
    # avoid overwriting a same-named file from a previous book
    counter = 1
    stem, suffix = stored_path.stem, stored_path.suffix
    while stored_path.exists():
        stored_path = pdf_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    with open(stored_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = Document(filename=file.filename, stored_path=str(stored_path), status="processing", uploaded_by=current_user.id)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        pages = extract_pdf_structure(str(stored_path))
        chunks = chunk_pages(pages)
        vector_store.build(doc.id, chunks)
        doc.page_count = len(pages)
        doc.status = "ready"
    except Exception as e:
        doc.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    # first successfully uploaded book becomes active automatically
    if not db.query(Document).filter(Document.is_active == True).first():  # noqa: E712
        doc.is_active = True

    db.commit()
    db.refresh(doc)
    return doc


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Document).order_by(Document.id.desc()).all()


@app.get("/documents/active", response_model=list[DocumentOut])
def get_active_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Document).filter(Document.is_active == True).order_by(Document.id.desc()).all()  # noqa: E712


@app.post("/documents/{doc_id}/toggle-active", response_model=DocumentOut)
def toggle_active_document(doc_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Toggles a book's active flag. Any number of books can be active
    at once; queries search across all of them together."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc or doc.status != "ready":
        raise HTTPException(status_code=404, detail="Document not found or not ready")

    doc.is_active = not doc.is_active
    db.commit()
    db.refresh(doc)

    active_ids = [d.id for d in db.query(Document).filter(Document.is_active == True).all()]  # noqa: E712
    vector_store.set_active_set(active_ids)

    return doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not settings.enable_upload_delete:
        raise HTTPException(status_code=403, detail="Deletion is disabled on this deployment")

    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    was_active = doc.is_active

    db.query(ChatMessage).filter(ChatMessage.document_id == doc_id).delete()
    db.delete(doc)
    db.commit()

    vector_store.delete(doc_id)
    Path(doc.stored_path).unlink(missing_ok=True)

    if was_active:
        fallback = db.query(Document).filter(Document.status == "ready").order_by(Document.id.desc()).first()
        if fallback:
            fallback.is_active = True
            db.commit()
            vector_store.load(fallback.id)

    return {"status": "deleted"}


@app.get("/documents/page/{page_number}")
def get_document_page_image(
    page_number: int,
    doc_id: int | None = None,
    token: str = Query(..., description="JWT access token (query param)"),
    db: Session = Depends(get_db),
):
    """Renders ONLY the requested page as a PNG image. If doc_id is
    omitted, uses the currently active document."""
    import fitz
    import io

    get_current_user_from_query_token(token, db)

    if doc_id is not None:
        doc_row = db.query(Document).filter(Document.id == doc_id).first()
    else:
        doc_row = db.query(Document).filter(Document.is_active == True).first()  # noqa: E712

    if not doc_row or doc_row.status != "ready":
        raise HTTPException(status_code=404, detail="No ready document available")

    pdf = fitz.open(doc_row.stored_path)
    if page_number < 1 or page_number > len(pdf):
        pdf.close()
        raise HTTPException(status_code=404, detail="Page number out of range")

    page = pdf[page_number - 1]
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    pdf.close()

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")