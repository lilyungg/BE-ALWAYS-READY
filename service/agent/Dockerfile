FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN hf download jinaai/jina-reranker-v2-base-multilingual
COPY . .

EXPOSE 8010