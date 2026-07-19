from typing import List, Optional
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    is_admin: bool

    class Config:
        from_attributes = True


class DocumentOut(BaseModel):
    id: int
    filename: str
    page_count: int
    status: str
    is_active: bool

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    doc_id: int
    filename: str
    page: int
    text: str
    score: float
class PageRef(BaseModel):
    doc_id: int
    filename: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    pages: List[PageRef]
    sources: List[SourceChunk]

class ChatHistoryItem(BaseModel):
    role: str
    content: str
    pages: Optional[str] = None

    class Config:
        from_attributes = True
