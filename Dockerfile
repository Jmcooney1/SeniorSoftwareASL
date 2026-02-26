# Use a Python base image
FROM python:3.10-slim

# Force root user for installations
USER root

# Install modern system dependencies for OpenCV/MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your requirements first for caching
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files
COPY . .

# This keeps the container alive
CMD ["python", "motion_acc_test.py"]