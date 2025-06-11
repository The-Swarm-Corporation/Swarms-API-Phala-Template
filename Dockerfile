FROM python:3.11-slim

# Set environment variables to ensure Python output is logged and bytecode is not written
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy the requirements file first for dependency caching
COPY api/requirements.txt .


# Install build dependencies, then install Python dependencies, and finally remove build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get purge -y --auto-remove gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Update swarms
RUN pip install -U swarms==7.8.4

# Copy the API source code into the container
COPY api/ .

# Create a non-root user and change ownership of the application folder
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser /app
USER appuser

# Expose port 80 for the application
EXPOSE 8080

# Start Uvicorn with production settings directly, set to 10 workers, and add additional settings
CMD ["uvicorn", "api:app", "--host=0.0.0.0", "--port=8080", "--workers=8", "--timeout-keep-alive=65", "--log-level=info", "--reload"]