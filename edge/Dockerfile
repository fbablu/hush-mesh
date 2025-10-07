FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY edge/ .

# Download YOLO model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLO model download failed, will use synthetic mode"

# Run the edge agent
CMD ["python", "agent.py"]