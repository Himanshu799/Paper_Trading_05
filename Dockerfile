# 1. Base and system deps for build
FROM python:3.10-slim AS builder
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Final image
FROM python:3.10-slim
WORKDIR /app

# 4. Copy installed packages
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 5. Copy your code + data
COPY deploy.py .
COPY models/ models/
COPY preprocessed_data/ preprocessed_data/
COPY ppo_ceemd_cnnlstm_rl.zip .

# 6. Entrypoint
CMD ["python", "deploy.py"]
