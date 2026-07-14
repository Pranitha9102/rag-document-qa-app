from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
from pypdf import PdfReader
import io

from database import get_db
from models import Document, Chunk
from chunking import chunk_text
from embeddings import get_embeddings

app = FastAPI()

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