import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import shutil
from typing import List
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma as LCChroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_service")
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
    raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")

LLM_MODEL_ALIAS = os.getenv("LLM_MODEL_ALIAS", "gemini-flash-latest")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Xenova/multi-qa-MiniLM-L6-cos-v1-quantized")
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "vector-db")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers")

llm = None
embedding_model = None
vector_store = None
retrieval_chain = None
app = FastAPI(title="RAG Service")

@app.on_event("startup")
def startup_event():
    global llm, embedding_model, vector_store, retrieval_chain

    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_ALIAS,
            google_api_key=GOOGLE_API_KEY,
            api_version="v1",
            temperature=0.2
        )
        logger.info(f"LLM 초기화 완료 (model={LLM_MODEL_ALIAS})")
    except Exception as e:
        logger.exception("LLM 초기화 실패")
        llm = None

    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
        logger.info(f"임베딩 모델 로드 완료 (model={EMBEDDING_MODEL_NAME})")
    except Exception as e:
        logger.exception("임베딩 모델 로드 실패")
        embedding_model = None

    try:
        chroma_client = chromadb.HttpClient(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)
        
        vector_store = LCChroma(
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_model,
        )
        
        logger.info(f"Chroma 벡터스토어 연결 완료: {VECTOR_DB_HOST}:{VECTOR_DB_PORT} / collection={COLLECTION_NAME}")
    except Exception as e:
        logger.exception("Chroma 연결 실패")
        vector_store = None

    if llm and vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        prompt = ChatPromptTemplate.from_template("""
        당신은 논문 분석 전문가입니다. [지시사항]과 [문맥]을 바탕으로만 질문에 답해야 합니다.
        [지시사항]
        1. 모든 문장은 15~25개의 단어(어절)로 구성해야 합니다.
        2. 답변은 반드시 아래의 [답변 형식]을 따라야 합니다.
        3. [문맥]에 근거가 없는 내용은 절대 답변하지 마세요.
        [답변 형식]
        **Answer:** (질문에 대한 명확한 대답)
        **Reasoning:** (해당 대답이 나온 이유 및 판단 근거)
        **Supplement:** (제공된 문맥을 바탕으로 한 보충 설명)
        [문맥]:
        {context}
        [질문]:
        {input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        logger.info("RAG chain 구성 완료")
    else:
        logger.error("RAG chain 구성 실패: LLM 또는 Vector Store 초기화 실패")
        retrieval_chain = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: List[str]

class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    status: str

@app.post("/upload")
async def upload_paper(request: Request):
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector DB에 연결할 수 없습니다.")

    filename = request.headers.get("X-Filename", "default_temp_file.pdf")
    temp_file_path = f"./{filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            async for chunk in request.stream():
                buffer.write(chunk)
        logger.info(f"파일 수신 완료: {filename}")

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        docs = text_splitter.split_documents(documents)

        if not docs:
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")

        logger.info(f"{len(docs)}개 조각을 Vector DB에 저장합니다...")
        vector_store.add_documents(docs)

        os.remove(temp_file_path)
        return {"filename": filename, "chunks_added": len(docs), "status": "성공적으로 처리되어 DB에 저장되었습니다."}

    except Exception as e:
        logger.exception("파일 처리 중 오류")
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_paper(request: QueryRequest):
    if retrieval_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain이 준비되지 않았습니다.")

    try:
        logger.info(f"질문 수신: {request.question!r}")
        response = await retrieval_chain.ainvoke({"input": request.question})

        answer = response.get("answer", "")
        raw_context: List[Document] = response.get("context", [])
        context_docs = [doc.page_content for doc in raw_context if hasattr(doc, "page_content")]

        return QueryResponse(
            question=request.question,
            answer=answer or "",
            context=context_docs or []
        )
    except Exception as e:
        logger.exception("질문 처리 중 오류")
        raise HTTPException(status_code=500, detail=f"질문 처리 중 오류 발생: {str(e)}")

@app.post("/clear_db")
async def clear_database():
    global vector_store
    logger.info("Vector DB 초기화 요청 수신...")
    try:
        if vector_store and hasattr(vector_store, "delete_collection"):
            vector_store.delete_collection()
            logger.info("Chroma 컬렉션 삭제 호출")

        chroma_client = chromadb.HttpClient(host=VECTOR_DB_HOST, port=VECTOR_DB_PORT)
        vector_store = LCChroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
        )
        return {"status": "벡터 DB가 성공적으로 초기화되었습니다."}
    except Exception as e:
        logger.exception("DB 초기화 중 오류")
        raise HTTPException(status_code=500, detail=f"DB 초기화 중 오류 발생: {str(e)}")

@app.get("/health")
def health_check():
    if retrieval_chain is not None and llm is not None and embedding_model is not None and vector_store is not None:
        return {"service_status": "READY"}
    else:
        raise HTTPException(status_code=503, detail="Service is still loading dependencies")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)