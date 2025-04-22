# Dockerfile for Django (backend)
FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /backend

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /backend

RUN pip install --no-cache-dir -r requirements.txt

COPY ./project_root /backend

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]