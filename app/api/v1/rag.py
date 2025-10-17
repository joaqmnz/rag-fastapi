from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
import io
from app.db.mongo import get_db
from app.schemas.rag import UploadDocResponse, AskRequest, AnswerResponse
from app.rag.store import upsert_document, search_similar_chunks
from app.llm.openai_client import embed_texts, chat_completion
from pypdf import PdfReader

router = APIRouter(tags=["rag"])

@router.post("/rag/documents", response_model=UploadDocResponse)
async def upload_document(
    db=Depends(get_db),
    # multipart com PDF
    file: UploadFile = File(...),
    title: str = Form(...),
):
    # Opcional: proteger com token -> current_user = Depends(get_current_user_id)

    # validações de PDF
    if file.content_type not in {"application/pdf", "application/x-pdf", "binary/octet-stream"} and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Envie um arquivo PDF válido")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Arquivo PDF vazio")

    # Extração de texto do PDF
    text = _extract_text_from_pdf_bytes(data)
    if not text or not text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Não foi possível extrair texto do PDF")

    res = await upsert_document(db, title, text)
    return UploadDocResponse(**res)


def _extract_text_from_pdf_bytes(data: bytes) -> str:
   
    # Tenta abrir o PDF e trata erros de leitura/arquivo inválido
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Falha ao ler PDF: {str(e)}",
        )

    pages = getattr(reader, "pages", [])
    # Limite de páginas: até 15
    try:
        page_count = len(pages)
    except Exception:
        page_count = 0
    if page_count > 15:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PDF excede o limite de 15 páginas")

    texts = []
    for page in pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            texts.append(t)
    return "\n\n".join(texts)


@router.post("/rag/ask", response_model=AnswerResponse)
async def ask(body: AskRequest, db=Depends(get_db)):
    # 1) embedding da pergunta
    q_emb = embed_texts([body.question])[0]
    
    # 2) busca chunks similares
    results = await search_similar_chunks(db, body.doc_id, q_emb, k=body.k)
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No context found")
    
    # 3) monta prompt com contexto
    context = "".join([r.get("chunk", "") for r in results])
    system = "Você é um assistente que responde com base apenas no CONTEXTO fornecido. Se não houver informação suficiente, diga que não sabe."
    user = f"CONTEXTO: {context} PERGUNTA: {body.question}"
    
    # 4) chama LLM
    answer = chat_completion(system, user)
    # Referencia as fontes pelos IDs dos chunks retornados
    sources = [f"chunk_id={str(r.get('_id'))}" for r in results]
    return AnswerResponse(answer=answer, sources=sources)
