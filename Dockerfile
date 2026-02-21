FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Turn off progress bar to prevent thread crashes
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

COPY . .

CMD ["python", "evaluate.py"]