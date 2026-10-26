FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
RUN useradd -m appuser && mkdir -p /data/chroma && chown -R appuser /data /app
USER appuser
ENV CHROMA_DIR=/data/chroma HOME=/home/appuser
EXPOSE 8001
# One worker: job state is held in memory and Chroma is a local store.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
