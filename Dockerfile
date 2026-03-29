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
    # Added X11/Qt libraries for the GUI window to work
    libxcb-xinerama0 \
    libqt5gui5 \
    libx11-xcb1 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy your requirements first for caching
COPY requirements.txt .

# Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

ENV DISPLAY=host.docker.internal:0.0

# Copy your project files
COPY . .

# This keeps the container alive
CMD ["sleep", "infinity"]