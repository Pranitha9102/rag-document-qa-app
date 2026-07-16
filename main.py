from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel
from pypdf import PdfReader
import io

from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
import os

from fastapi.responses import StreamingResponse
import json

from database import get_db
from models import Document, Chunk
from chunking import chunk_text
from embeddings import get_embeddings, get_embedding

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = "gpt-4o-mini"

@app.get("/")
def read_root():
    return {"message": "RAG app is alive"}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Extract text from PDF
    pdf_bytes = file.file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    # 2. Chunk the text
    chunks = chunk_text(text)

    # 3. Embed all chunks in one batch
    embeddings = get_embeddings(chunks)

    # 4. Save document
    document = Document(filename=file.filename)
    db.add(document)
    db.commit()
    db.refresh(document)

    # 5. Save each chunk with its embedding
    for chunk_content, embedding in zip(chunks, embeddings):
        chunk = Chunk(
            document_id=document.id,
            content=chunk_content,
            embedding=embedding
        )
        db.add(chunk)

    db.commit()

    return {
        "document_id": document.id,
        "filename": document.filename,
        "num_chunks": len(chunks)
    }


class Question(BaseModel):
    query: str
    top_k: int = 5


RELEVANCE_THRESHOLD = 0.6


@app.post("/ask")
def ask_question(question: Question, db: Session = Depends(get_db)):
    # 1. Embed the question
    query_embedding = get_embedding(question.query)

    # 2. Vector search
    stmt = (
        select(Chunk, Chunk.embedding.cosine_distance(query_embedding).label("distance"))
        .order_by("distance")
        .limit(question.top_k)
    )
    results = db.execute(stmt).all()

    # 3. Relevance threshold
    if not results or results[0].distance > RELEVANCE_THRESHOLD:
        return {
            "query": question.query,
            "answer": "I don't have information about that in the uploaded documents.",
            "citations": []
        }

    # 4. Build the context block from retrieved chunks
    context_parts = []
    citations = []
    for i, (chunk, distance) in enumerate(results, start=1):
        context_parts.append(f"[Source {i}]\n{chunk.content}")
        citations.append({
            "source_number": i,
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "distance": float(distance)
        })
    context = "\n\n".join(context_parts)

    # 5. Prompt the LLM
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided sources. Cite each fact you use with [Source N]. "
        "If the sources don't contain the answer, say so."
    )
    user_prompt = f"Sources:\n{context}\n\nQuestion: {question.query}"

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    answer = response.choices[0].message.content

    return {
        "query": question.query,
        "answer": answer,
        "citations": citations
    }

@app.post("/ask/stream")
def ask_question_stream(question: Question, db: Session = Depends(get_db)):
    # 1. Embed the question
    query_embedding = get_embedding(question.query)

    # 2. Vector search
    stmt = (
        select(Chunk, Chunk.embedding.cosine_distance(query_embedding).label("distance"))
        .order_by("distance")
        .limit(question.top_k)
    )
    results = db.execute(stmt).all()

    # 3. Threshold refusal (non-streaming)
    if not results or results[0].distance > RELEVANCE_THRESHOLD:
        refusal_msg = "I don't have information about that in the uploaded documents."
        def refusal():
            yield f"data: {json.dumps({'type': 'answer', 'content': refusal_msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(refusal(), media_type="text/event-stream")

    # 4. Build context and citations
    context_parts = []
    citations = []
    for i, (chunk, distance) in enumerate(results, start=1):
        context_parts.append(f"[Source {i}]\n{chunk.content}")
        citations.append({
            "source_number": i,
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "distance": float(distance)
        })
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided sources. Cite each fact you use with [Source N]. "
        "If the sources don't contain the answer, say so."
    )
    user_prompt = f"Sources:\n{context}\n\nQuestion: {question.query}"

    def event_stream():
        # First event: send the citations up front
        yield f"data: {json.dumps({'type': 'citations', 'content': citations})}\n\n"

        # Then: stream the LLM tokens
        stream = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Final: signal done
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")