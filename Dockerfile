FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FL_SERVER_ADDRESS=0.0.0.0:8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY launch.py server.py client.py train_centralized.py evaluate.py api.py predict.py ./
COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p models results logs

EXPOSE 8080 5000

CMD ["python", "launch.py", "--mode", "server"]
