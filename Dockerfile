# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies with pip cache mount
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

RUN chmod +x ./prod_setup.sh
RUN chmod +x ./entrypoint.sh

# path to nltk data
ENV NLTK_DATA=/app/nltk_data

# Expose port (Railway auto-detects 8080 by default)
EXPOSE 8080

# Start command
CMD ["./entrypoint.sh"]
