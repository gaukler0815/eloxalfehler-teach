FROM python:3.11-slim

WORKDIR /app

# Abhängigkeiten
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Anwendung
COPY backend/ backend/
COPY frontend/ frontend/

# Persistente Daten (Datenbank, Uploads, VAPID-Schlüssel)
ENV FK_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8000
WORKDIR /app/backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
