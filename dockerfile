# -------------------------------------------------------
# Base Image
# -------------------------------------------------------
FROM python:3.12-slim

# -------------------------------------------------------
# Environment Variables
# -------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -------------------------------------------------------
# Set Working Directory
# -------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------
# Create non-root user for security hardening
# -------------------------------------------------------
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/sh -m appuser

# -------------------------------------------------------
# Copy requirements first (better layer caching)
# -------------------------------------------------------
COPY requirements.txt .

# -------------------------------------------------------
# Install Python dependencies
# -------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------
# Copy the entire project and transfer ownership
# -------------------------------------------------------
COPY . .
RUN chown -R appuser:appuser /app

# -------------------------------------------------------
# Switch to non-root user
# -------------------------------------------------------
USER appuser

# -------------------------------------------------------
# Expose port and Default command
# -------------------------------------------------------
EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]