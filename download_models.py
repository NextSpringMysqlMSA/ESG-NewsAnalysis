import os
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 다운로드할 모델 목록 (modelkafka.py와 일치)
MODEL_LIST = {
    "translation": {
        "name": os.environ.get("TRANSLATE_MODEL_NAME", "facebook/nllb-200-distilled-600M"),
        "type": "seq2seq"
    },
    "sentiment": {
        "name": os.environ.get("SENTIMENT_MODEL_NAME", "snunlp/KR-FinBert-SC"),
        "type": "pipeline",
        "task": "sentiment-analysis"
    },
    "esg": {
        "name": os.environ.get("ESG_MODEL_NAME", "yiyanghkust/finbert-esg"),
        "type": "pipeline",
        "task": "text-classification"
    }
}

# Hugging Face 캐시 디렉토리 설정 (선택 사항, Docker 빌드 시 기본 경로 사용)
# TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE")
# HF_HOME = os.environ.get("HF_HOME")

def download_model(model_info):
    model_name = model_info["name"]
    model_type = model_info["type"]
    logger.info(f"Downloading {model_type} model: {model_name}...")
    try:
        if model_type == "seq2seq":
            AutoTokenizer.from_pretrained(model_name)
            AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif model_type == "pipeline":
            task = model_info["task"]
            pipeline(task, model=model_name, tokenizer=model_name)
        logger.info(f"Successfully downloaded {model_name}.")
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}", exc_info=True)
        # 빌드 실패를 유도하기 위해 예외를 다시 발생시킬 수 있음
        # raise

if __name__ == "__main__":
    logger.info("Starting model download process...")
    
    # 캐시 디렉토리 정보 출력 (디버깅용)
    # logger.info(f"TRANSFORMERS_CACHE: {TRANSFORMERS_CACHE}")
    # logger.info(f"HF_HOME: {HF_HOME}")
    # logger.info(f"Default cache (HF): {os.path.expanduser('~/.cache/huggingface/hub')}")
    # logger.info(f"Default cache (Transformers): {os.path.expanduser('~/.cache/huggingface/transformers')}")

    for model_key, info in MODEL_LIST.items():
        download_model(info)
    
    logger.info("Model download process finished.") 