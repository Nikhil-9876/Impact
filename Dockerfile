FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .

# Copy model files if they exist (optional)
COPY best_model.pt* ./
COPY optimal_threshold.txt* ./

# Expose FastAPI port (Hugging Face Spaces expects 7860)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
