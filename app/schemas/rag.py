from pydantic import BaseModel, Field
from typing import List

class UploadDocRequest(BaseModel):
    title: str
    content: str

class UploadDocResponse(BaseModel):
    doc_id: str
    chunks: int

class AskRequest(BaseModel):
    doc_id: str
    question: str
    k: int = 5

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]