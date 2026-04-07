FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && rm -rf /var/lib/apt/lists/*

# Python dependencies (copy from project root, not server/)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project (run docker build from datacleanenv/ root)
COPY . .

# Create reports output directory
RUN mkdir -p batch_cleaner/reports

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
