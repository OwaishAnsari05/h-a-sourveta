FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/root/.cache/huggingface
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV OCR_LANGUAGES=eng+hin+mar

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-mar \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY agents ./agents
COPY evaluation ./evaluation
COPY generation ./generation
COPY ingestion ./ingestion
COPY vectorstore ./vectorstore
RUN mkdir -p data/documents data/chroma_db data/chunks

EXPOSE 8000
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
