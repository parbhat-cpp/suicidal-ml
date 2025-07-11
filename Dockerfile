# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies with pip cache mount
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

CMD chmod ./prod_setup.sh && ./prod_setup.sh

# Expose port (Railway auto-detects 8080 by default)
EXPOSE 8080

# Start command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
