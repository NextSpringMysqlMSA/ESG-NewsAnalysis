FROM python:3.12-slim

# 필수 패키지
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && apt-get clean

# Python 패키지 설치
RUN pip install --upgrade pip
RUN pip install torch transformers kafka-python

# 작업 디렉토리
WORKDIR /app
COPY . /app

CMD ["python", "modelkafka.py"]
