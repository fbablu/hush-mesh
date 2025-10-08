FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mission_planner.py .

EXPOSE 8000

# Human-in-the-loop safety check
ENV REQUIRE_HUMAN_APPROVAL=true
ENV MAX_THREAT_RESPONSE_LEVEL=defensive_only

CMD ["python", "mission_planner.py"]