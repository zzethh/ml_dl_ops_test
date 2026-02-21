# Use a lightweight Python base image
FROM python:3.9-slim-bullseye

# Set working directory
WORKDIR /app

# Install git (needed for some HF dependencies)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (Turned off progress bar to keep build logs clean)
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Copy the source code
COPY src/ ./src/

# Default environment variables
ENV HF_REPO_ID="Zenith754/STL10-ResNet18"
# Epochs turned to 3 as requested!
ENV EPOCHS="3" 

# Run the training script, passing the environment variables as arguments
CMD ["sh", "-c", "python src/train.py --repo_id $HF_REPO_ID --epochs $EPOCHS"]