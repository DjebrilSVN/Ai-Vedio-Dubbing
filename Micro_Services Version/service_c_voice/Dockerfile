# 1. Use the official PyTorch GPU image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
# 2. Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg git

# 3. Set the working directory
WORKDIR /app

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Orchestrator / UI code
COPY app.py .

# 6. Open Port 7860 for Gradio
EXPOSE 7860

# 7. Start the Web UI
CMD ["python", "app.py"]