# Multi-platform Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    FLASK_ENV=production

# Install system dependencies required for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install python dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository code
COPY . /workspace/

# Expose server port
EXPOSE 5000

# Start production server with gunicorn
CMD ["gunicorn", "-w", "2", "--threads", "4", "-b", "0.0.0.0:5000", "app.server:app"]
