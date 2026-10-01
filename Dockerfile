FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uvicorn fastapi httpx

COPY main.py /app/main.py

ENV PORT=6060
EXPOSE 6060

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6060"]
