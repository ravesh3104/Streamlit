# Industrial Equipment Failure Prediction — dashboard image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# System libs needed by matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && rm -rf /var/lib/apt/lists/*

# Install dependencies first so the layer caches across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + bundled data + pre-trained artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

# Healthcheck hits Streamlit's own health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health').read()" || exit 1

# Bind to 0.0.0.0 so the app is reachable from outside the container
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
