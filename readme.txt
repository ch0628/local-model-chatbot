1. 라즈베리 파이용 경량 모델
지난번에 추천해 드린 모델이 바로 라즈베리 파이를 위한 최적의 시작점입니다.

모델명: skt/kogpt2-base-v2

선정 이유:

한국어 특화: 한국어 데이터를 기반으로 만들어져 한국어 대화 능력이 좋습니다.

경량: 파라미터 수가 약 1.25억 개로, 라즈베리 파이 8GB RAM 환경에서 단독으로 실행해 볼 만한 현실적인 마지노선에 가깝습니다.

안정성: 널리 사용되고 검증된 모델이라 관련 자료를 찾기 쉽습니다.

이 모델로 먼저 챗봇 서버를 완성하고 라즈베리 파이에서 성능을 테스트해보는 것이 가장 좋습니다.

2. 토크나이저(Tokenizer)의 정체 📖
토크나이저는 AI 모델의 '번역가' 또는 **'사전'**이라고 생각하시면 됩니다.

AI 모델(컴퓨터)은 '안녕하세요' 같은 인간의 언어를 직접 이해하지 못합니다. 오직 숫자만 이해할 수 있죠. 토크나이저는 이 둘 사이를 이어주는 역할을 합니다.

작동 방식:

토큰화 (Tokenization): 사람이 문장을 입력하면("안녕하세요") → 토크나이저는 그 문장을 의미 있는 조각(토큰)으로 나눕니다 (["안녕", "하세", "요"]).

정수 인코딩 (Integer Encoding): 토크나이저는 각 조각을 자신의 '사전'에서 찾아 고유 번호로 바꿉니다 ([8001, 8010, 8023]).

모델 처리: AI 모델은 이 숫자 배열을 입력받아 계산을 수행하고, 결과로 새로운 숫자 배열을 출력합니다 ([9012, 9034, 9056]).

디코딩 (Decoding): 토크나이저는 모델이 출력한 숫자 배열을 다시 '사전'을 이용해 사람이 읽을 수 있는 문장으로 번역합니다 ("반갑습니다").

"지난번과 왜 다른가요?"
지난번에 pipeline("text-generation", model="distilgpt2") 라고 썼을 때도, 사실 내부적으로는 해당 모델과 짝이 맞는 토크나이저가 자동으로 로드되어 사용되었습니다. pipeline 함수가 편리하게 이 과정을 숨겨줬던 것뿐입니다.

이번에 AutoTokenizer.from_pretrained(...) 처럼 코드를 명시적으로 작성한 이유는, 모델과 토크나이저를 직접 제어하여 나중에 더 세밀한 설정(예: 특정 단어 추가 등)을 하거나 코드의 작동 방식을 더 명확하게 만들기 위해서입니다.

3. ai/gpt-oss 이미지를 사용하지 않는 이유
ai/gpt-oss 같은 이미지는 '모든 것이 갖춰진 개발 환경' 또는 **'데모용 이미지'**에 가깝습니다. 우리가 직접 만들고 있는 **'최적화된 맞춤형 서버'**와는 목적이 다릅니다.

마치 **"가구가 모두 딸린 풀옵션 원룸(ai/gpt-oss)"**과 **"내가 직접 설계해서 짓는 나만의 집(현재 프로젝트)"**의 차이와 같습니다.

우리가 ai/gpt-oss를 사용하지 않는 이유는 다음과 같습니다.

목적 불일치: ai/gpt-oss는 특정 AI 모델을 쉽게 테스트해보거나 Jupyter Notebook 같은 연구 환경을 제공하는 것이 목적입니다. 우리가 만들려는 가볍고 빠른 FastAPI API 서버와는 목적이 다릅니다.

불필요한 기능 및 크기: 풀옵션 원룸처럼, 우리 프로젝트에 필요 없는 수많은 라이브러리와 도구들이 미리 설치되어 있어 이미지 크기가 매우 큽니다. 이는 라즈베리 파이 환경에 치명적입니다. 우리는 python:3.9-slim이라는 빈 집에 우리에게 꼭 필요한 최소한의 가구만 들여놓는 방식으로 만들고 있습니다.

아키텍처 문제: 해당 이미지는 대부분 PC용 CPU(linux/amd64)를 기준으로 만들어졌을 가능성이 높습니다. 우리는 라즈베리 파이(linux/arm64)에서도 동작해야 하므로, 처음부터 두 아키텍처를 모두 지원하는 python 공식 이미지를 기반으로 직접 만드는 것이 훨씬 효율적입니다.

통제권 및 최적화: 직접 이미지를 만들면 모든 라이브러리의 버전을 관리하고, 설정을 최적화하는 등 완벽한 통제권을 가질 수 있습니다. 이는 안정적인 서버를 만드는 데 필수적입니다.




distilgpt2 사용 모델 
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import pipeline
from contextlib import asynccontextmanager
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "distilgpt2"

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading {MODEL_NAME}...")
    try:
        app.state.generator = pipeline("text-generation", model=MODEL_NAME)
        logger.info(f"{MODEL_NAME} 로드 완료")
    except Exception as e:
        logger.error(f"모델 {MODEL_NAME} 로드 실패: {e}")
        app.state.generator = None

    yield
    logger.info("서버 종료 중..., 모델 메모리 정리 시작")
    if hasattr(app.state, "generator"):
        del app.state.generator
        logger.info("모델 메모리 정리 완료")


app = FastAPI(
    title="생성형ai_sub1",
    description="Hugging Face의 distilgpt2 모델을 사용한 텍스트 생성 API",
    version="1.0.0",
    lifespan=lifespan,
)

class PromptRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.8
    do_sample: bool = True

class GeneratedResponse(BaseModel):
    input_prompt: str
    generated_text: str


@app.post("/generate", response_model=GeneratedResponse)
async def generate_text(request: PromptRequest, req: Request):
    generator = getattr(req.app.state, "generator", None)

    if generator is None:
        raise HTTPException(status_code=503, detail="AI 모델이 아직 로드되지 않았습니다. 잠시 후 다시 시도해 주세요.")

    logger.info(f"입력 프롬프트: {request.prompt}")

    try:
        outputs = generator(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            num_return_sequences=1,
        )
        generated_text = outputs[0]["generated_text"].strip()

        return GeneratedResponse(
            input_prompt=request.prompt,
            generated_text=generated_text,
        )

    except Exception as e:
        logger.error(f"텍스트 생성 중 오류 발생: {e}")
        return GeneratedResponse(
            input_prompt=request.prompt,
            generated_text=f"텍스트 생성 중 오류가 발생했습니다: {str(e)}",
        )


@app.get("/")
def read_root(req: Request):
    """서비스 상태 확인 (Health Check)"""
    status = "READY" if getattr(req.app.state, "generator", None) else "LOADING"
    return {"service_status": status, "model": MODEL_NAME}
