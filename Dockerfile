FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ffmpeg git wget curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Security Override for Hugging Face Spaces (Spaces runs containers as generic non-root users)
RUN mkdir -p /app/data/models_cache && chmod -R 777 /app

# Native Redirection: Force HuggingFace and PyTorch to cache within our 777 zone instead of the locked /root/
ENV HF_HOME=/app/data/models_cache
ENV TORCH_HOME=/app/data/models_cache
ENV XDG_CACHE_HOME=/app/data/models_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
# Final permission blanket to ensure all deployed files are writable by HF's user
RUN chmod -R 777 /app

EXPOSE 7860
CMD ["python", "app.py"]
