# Water Analysis Framework Docker Environment
# Based on Python 3.10 with scientific computing stack

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    libhdf5-dev \
    libnetcdf-dev \
    libgdal-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY test.py .
COPY water_pixels_analysis.py .
COPY cloud_water_correlation.py .
COPY README.md .
COPY LICENSE .

# Create directories for outputs
RUN mkdir -p ndwi_images images logs

# Set default command
CMD ["python", "test.py"] 