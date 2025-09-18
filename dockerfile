FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both app/ and logger.py
COPY app app
COPY logger.py .


CMD ["bash", "-c", "python logger.py && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4"]

EXPOSE 8000