FROM python:3.11-slim

WORKDIR /app

# Install build tools needed for llama-cpp-python (CMake + C++ compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pre-set CMAKE flags for CPU-only build (no CUDA/AVX2 assumptions on free tier)
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
ENV FORCE_CMAKE=1

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

EXPOSE 7860

CMD ["python", "app.py"]