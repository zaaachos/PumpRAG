# temp stage
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl

# Install python dependencies
COPY pyproject.toml .
COPY poetry.lock .
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    pip install --upgrade pip && \
    pip install poetry && \
    poetry install

COPY . .

# final stage
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app .

ENTRYPOINT ["./entrypoint.sh"]
