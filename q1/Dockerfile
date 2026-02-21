FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

COPY . .

CMD ["python", "evaluate.py"]