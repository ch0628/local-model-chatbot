import os
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging

logging.basicConfig(
    format = "%(asctime)s %(levelname)s: %(message)s",
    level = logging.INFO,
    datefmt = "%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "skt/kogpt2-base-v2")
DB_PATH = "chat.db"

def init_db():
    logger.info("데이터베이스를 초기화합니다...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 대화 기록을 저장할 테이블 생성
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logger.info("데이터베이스 초기화 완료.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info(f"'{MODEL_NAME}' 로드 중 ... ")
    try:
        # 모델 및 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        app.state.generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        logger.info(f"{MODEL_NAME} 로드 완료")
    except Exception as e:
        logger.error(f"{MODEL_NAME} 로드 실패: {e}")
        app.state.generator = None
    yield
    logger.info("서버 종료 중..., 모델 메모리 정리 시작...")
    if hasattr(app.state, "generator"):
        del app.state.generator
        logger.info("모델 메모리 정리 완료")

app = FastAPI(
    title="챗봇 서버",
    description="Hugging Face의 kogpt2 모델을 사용한 텍스트 생성 API",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory="static"), name="static")

class UserRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7 
    do_sample: bool = True
    repetition_penalty: float = 1.3 
    top_p: float = 0.9
    top_k: int = 50 

class BotResponse(BaseModel):
    input_prompt: str
    generated_text: str

def build_prompt_with_history(user_input: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM history ORDER BY timestamp DESC LIMIT 3")
    rows = cursor.fetchall()
    conn.close()

    history = [f"{row[0]}: {row[1]}" for row in reversed(rows)] # 순서를 다시 뒤집어 시간순으로
    history_str = "\n".join(history)

    system_prompt = "You are a helpful AI assistant named 'IronBot'. You must answer concisely and only in natural Korean. Do not generate English words or repetitive phrases."
    
    return f"{system_prompt}\n\n{history_str}\nHuman: {user_input}\nAI:"


@app.post("/chat", response_model=BotResponse)
async def chat(request: UserRequest):
    if app.state.generator is None:
        raise HTTPException(status_code=503, detail="AI 모델이 아직 준비되지 않았습니다.")

    conversation_prompt = build_prompt_with_history(request.prompt)
    logger.info(f"모델에 전달할 최종 프롬프트:\n---\n{conversation_prompt}\n---")

    try:
        outputs = app.state.generator(
            conversation_prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty,
            top_p=request.top_p,
            top_k=request.top_k, 
            num_return_sequences=1,
            eos_token_id=app.state.generator.tokenizer.eos_token_id
        )
        
        raw_generated_text = outputs[0]["generated_text"]
        
        split_result = raw_generated_text.rsplit("AI:", 1)

        if len(split_result) > 1:
            ai_answer = split_result[1].strip()
        else:
            ai_answer = raw_generated_text.replace(conversation_prompt, "").strip()
            
        if not ai_answer and raw_generated_text:
             logger.warning("AI 답변 추출에 실패하여 원본 텍스트를 사용합니다.")
             ai_answer = raw_generated_text.strip()
    
        logger.info(f"AI 응답: {ai_answer}")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO history (role, content) VALUES (?, ?)", ("Human", request.prompt))
        cursor.execute("INSERT INTO history (role, content) VALUES (?, ?)", ("AI", ai_answer))
        conn.commit()
        conn.close()

        return BotResponse(
            input_prompt=request.prompt,
            generated_text=ai_answer,
        )
    except Exception as e:
        logger.error(f"텍스트 생성 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"텍스트 생성 오류: {e}")

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.get("/health")
def health_check():
    status = "READY" if app.state.generator else "LOADING"
    return {"service_status": status, "model": MODEL_NAME}
