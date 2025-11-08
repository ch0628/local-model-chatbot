import os
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import List
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_gateway")

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8000")

app = FastAPI(title="API Gateway")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: List[str]

class UploadResponse(BaseModel):
    filename: str
    status: str

# --- 파일 스트리밍 헬퍼 ---
async def _file_streamer(file: UploadFile):
    await file.seek(0)
    chunk = await file.read(8192)
    while chunk:
        yield chunk
        chunk = await file.read(8192)

@app.post("/upload", response_model=UploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    logger.info(f"'/upload' 요청 수신: {file.filename}")
    headers = {'X-Filename': file.filename}
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{RAG_SERVICE_URL}/upload",
                content=_file_streamer(file),
                headers=headers
            )
            response.raise_for_status() 
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"RAG Service 통신 오류 (upload): {e}")
            raise HTTPException(status_code=502, detail=f"RAG 서비스 통신 오류: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"RAG Service 오류 (upload): {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.json())

@app.post("/query", response_model=QueryResponse)
async def query_paper(request: QueryRequest):
    logger.info(f"'/query' 요청 수신: {request.question}")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(f"{RAG_SERVICE_URL}/query", json=request.dict())
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"RAG Service 통신 오류 (query): {e}")
            raise HTTPException(status_code=502, detail=f"RAG 서비스 통신 오류: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"RAG Service 오류 (query): {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.json())

@app.post("/clear_db")
async def clear_database():
    logger.info("'/clear_db' 요청 수신")
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(f"{RAG_SERVICE_URL}/clear_db")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"RAG Service 통신 오류 (clear_db): {e}")
            raise HTTPException(status_code=502, detail=f"RAG 서비스 통신 오류: {e}")

@app.get("/health")
def health_check():
    return {"service_status": "READY"}