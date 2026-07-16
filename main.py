from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI
import io
import os
import json
import time

from database import get_db
from models import Document, Chunk
from chunking import chunk_text
from embeddings import get_embeddings, get_embedding
from metrics import (
    retrieval_latency,
    tokens_used,
    cost_dollars,
    queries_total,
    metrics_endpoint,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = "gpt-4o-mini"
RELEVANCE_THRESHOLD = 0.6

PROMPT_COST_PER_1M = 0.15
COMPLETION_COST_PER_1M = 0.60


@app.get("/")
def read_root():
    return {"message": "RAG app is alive"}


@app.get("/metrics")
def metrics():
    data, content_type = metrics_endpoint()
    return Response(content=data, media_type=content_type)


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    pdf_bytes = file.file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    document = Document(filename=file.filename)
    db.add(document)
    db.commit()
    db.refresh(document)

    for chunk_content, embedding in zip(chunks, embeddings):
        chunk = Chunk(document_id=document.id, content=chunk_content, embedding=embedding)
        db.add(chunk)
    db.commit()

    return {"document_id": document.id, "filename": document.filename, "num_chunks": len(chunks)}


class Question(BaseModel):
    query: str
    top_k: int = 5


@app.post("/ask")
def ask_question(question: Question, db: Session = Depends(get_db)):
    query_embedding = get_embedding(question.query)

    start = time.perf_counter()
    stmt = (
        select(Chunk, Chunk.embedding.cosine_distance(query_embedding).label("distance"))
        .order_by("distance")
        .limit(question.top_k)
    )
    results = db.execute(stmt).all()
    retrieval_latency.observe(time.perf_counter() - start)

    if not results or results[0].distance > RELEVANCE_THRESHOLD:
        queries_total.labels(outcome="refused").inc()
        return {
            "query": question.query,
            "answer": "I don't have information about that in the uploaded documents.",
            "citations": [],
        }

    context_parts = []
    citations = []
    for i, (chunk, distance) in enumerate(results, start=1):
        context_parts.append(f"[Source {i}]\n{chunk.content}")
        citations.append({
            "source_number": i,
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "distance": float(distance),
        })
    context = "\n\n".join(context_parts)

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
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content

    usage = response.usage
    tokens_used.labels(kind="prompt").inc(usage.prompt_tokens)
    tokens_used.labels(kind="completion").inc(usage.completion_tokens)
    cost = (
        usage.prompt_tokens * PROMPT_COST_PER_1M / 1_000_000
        + usage.completion_tokens * COMPLETION_COST_PER_1M / 1_000_000
    )
    cost_dollars.inc(cost)
    queries_total.labels(outcome="answered").inc()

    return {"query": question.query, "answer": answer, "citations": citations}


@app.post("/ask/stream")
def ask_question_stream(question: Question, db: Session = Depends(get_db)):
    query_embedding = get_embedding(question.query)

    start = time.perf_counter()
    stmt = (
        select(Chunk, Chunk.embedding.cosine_distance(query_embedding).label("distance"))
        .order_by("distance")
        .limit(question.top_k)
    )
    results = db.execute(stmt).all()
    retrieval_latency.observe(time.perf_counter() - start)

    if not results or results[0].distance > RELEVANCE_THRESHOLD:
        queries_total.labels(outcome="refused").inc()
        refusal_msg = "I don't have information about that in the uploaded documents."

        def refusal():
            yield f"data: {json.dumps({'type': 'answer', 'content': refusal_msg})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(refusal(), media_type="text/event-stream")

    context_parts = []
    citations = []
    for i, (chunk, distance) in enumerate(results, start=1):
        context_parts.append(f"[Source {i}]\n{chunk.content}")
        citations.append({
            "source_number": i,
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "distance": float(distance),
        })
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided sources. Cite each fact you use with [Source N]. "
        "If the sources don't contain the answer, say so."
    )
    user_prompt = f"Sources:\n{context}\n\nQuestion: {question.query}"

    def event_stream():
        yield f"data: {json.dumps({'type': 'citations', 'content': citations})}\n\n"
        stream = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        queries_total.labels(outcome="answered").inc()
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")