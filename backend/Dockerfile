FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY backend/requirements.txt .
COPY requirements.txt ./root_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r root_requirements.txt

# Copy application code
COPY backend/ .
COPY sim_configs/ ./sim_configs/
COPY models/ ./models/

EXPOSE 8000

# Human-in-the-loop safety check
ENV REQUIRE_HUMAN_APPROVAL=true
ENV MAX_THREAT_RESPONSE_LEVEL=defensive_only
ENV MQTT_BROKER=mosquitto
ENV MQTT_PORT=1883
ENV USE_MINIMAL=true

CMD ["python", "app.py"]
