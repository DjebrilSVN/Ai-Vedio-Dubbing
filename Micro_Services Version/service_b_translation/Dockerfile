# 1. Use the official PyTorch GPU image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
# 2. Set the working directory
WORKDIR /app

# 3. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the translation server code
COPY app.py .

# 5. Open Port 8002 for the Translation API
EXPOSE 8002

# 6. Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]