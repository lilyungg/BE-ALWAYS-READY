FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔥 Предзагрузка модели (кэш HuggingFace)
RUN hf download intfloat/multilingual-e5-large

COPY . .

EXPOSE 8000