# -------------------------------------------------------
# Base Image
# -------------------------------------------------------
FROM python:3.12-slim

# -------------------------------------------------------
# Prevent Python from writing .pyc files
# -------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1

# -------------------------------------------------------
# Disable output buffering
# -------------------------------------------------------
ENV PYTHONUNBUFFERED=1

# -------------------------------------------------------
# Set Working Directory
# -------------------------------------------------------
WORKDIR /app

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
# Copy the entire project
# -------------------------------------------------------
COPY . .

# -------------------------------------------------------
# Default command
# -------------------------------------------------------
#CMD ["python", "-m", "training.train", "--config", "configs/smoke_train.yaml"] -- training pipeline


CMD ["sh", "-c", "python -m uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]