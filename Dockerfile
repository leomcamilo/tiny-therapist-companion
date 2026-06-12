FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools for llama-cpp-python (CMake + C++ compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pre-set CMAKE flags for CPU-only build
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
ENV FORCE_CMAKE=1

# Install Python dependencies (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime-only dependencies (OpenBLAS for llama.cpp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# HuggingFace Spaces runtime env
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV BACKEND=llama_cpp
ENV MODEL_ID=nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1
ENV GGUF_MODEL_ID=bartowski/Llama-3.1-Nemotron-Nano-4B-v1.1-GGUF
ENV GGUF_FILENAME=*Q4_K_M.gguf
ENV HF_SPACE=yes

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

EXPOSE 7860

CMD ["python", "app.py"]