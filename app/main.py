from fastapi import FastAPI
from app.api.v1.rag import router as rag_router

app = FastAPI(title="RAG With FastAPI")
app.include_router(rag_router, prefix='/api/v1')