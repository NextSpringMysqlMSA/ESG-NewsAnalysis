# Python 버전 명시 (slim 버전 권장)
FROM python:3.12-slim

# 시스템 환경 변수 설정 (옵션)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Hugging Face 캐시 디렉토리 설정 (Docker 빌드 시 이 경로에 모델 다운로드)
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV HF_HOME=/app/.cache/huggingface

# 작업 디렉토리 설정
WORKDIR /app

# 필수 시스템 패키지 설치 (최소화)
# 모델 다운로드에 필요할 수 있는 curl, git 등은 필요시 주석 해제
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl \
#     git \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip

# 의존성 파일 복사 및 설치 (빌드 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 다운로드 스크립트 복사 및 실행
COPY download_models.py .
# 실행 권한 부여 (필요시)
# RUN chmod +x download_models.py 
RUN python download_models.py

# 나머지 소스 코드 복사
COPY . .

# 실행 명령어
CMD ["python", "modelkafka.py"]

# 주석: GPU를 사용하려면 아래와 같이 베이스 이미지 및 torch 설치를 변경해야 합니다.
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# ... (apt-get으로 python3, python3-pip 등을 설치하는 과정 필요)
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 그 외 requirements.txt의 내용은 동일하게 사용 가능 (torch 제외)