FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN pip install --no-cache-dir streamlit==1.63.0 requests==2.34.2 websocket-client==1.9.0

COPY frontend/app.py ./app.py

EXPOSE 8501

CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]