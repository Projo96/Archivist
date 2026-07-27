# API service image.
#
# Why a Dockerfile instead of `pip install` in the compose command:
# installing dependencies on every container start is slow, depends on
# the network and makes startup non-reproducible. Here dependencies sit
# in a cached layer and the container starts in a second.
FROM python:3.11-slim

# No .pyc files, unbuffered stdout: uvicorn logs show up immediately
# in `docker logs -f` instead of arriving in blocks.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first, on their own: when only the code changes,
# Docker reuses the dependency layer and reinstalls nothing.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ /app/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
