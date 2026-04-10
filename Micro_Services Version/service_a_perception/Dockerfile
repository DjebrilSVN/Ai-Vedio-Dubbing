# 1. Use the official PyTorch GPU image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
# 2. Install FFmpeg (Crucial for video/audio processing)
RUN apt-get update && apt-get install -y ffmpeg git

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy requirements and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the FastAPI server code
COPY app.py .

# 6. Open Port 8001 for the API
EXPOSE 8001

# 7. Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]