FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for audio processing and ML models
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directory for audio files
RUN mkdir -p /tmp/audio

# Expose port 10000 (Render's default)
EXPOSE 10000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
