FROM python:3.13-slim
WORKDIR /workspace
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .
CMD ["python", "scripts/verify_repository.py"]
