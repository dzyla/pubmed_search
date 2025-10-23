FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy entrypoint files; compose mounts full repo at runtime for dev
COPY pbmss_app.py /app/pbmss_app.py
COPY config_mss.yaml /app/config_mss.yaml

EXPOSE 8501

CMD ["streamlit", "run", "pbmss_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
