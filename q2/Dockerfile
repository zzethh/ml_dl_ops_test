FROM python:3.9-slim-bullseye


WORKDIR /app


RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir --progress-bar off -r requirements.txt


COPY src/ ./src/


ENV HF_REPO_ID="Zenith754/STL10-ResNet18"

ENV EPOCHS="3" 

CMD ["sh", "-c", "python src/train.py --repo_id $HF_REPO_ID --epochs $EPOCHS"]