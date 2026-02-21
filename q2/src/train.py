import torch
import torch.nn as nn
from torchvision import models
import wandb
from huggingface_hub import HfApi
import argparse
import os

# Import our updated functions from utils.py
from utils import get_dataloaders, evaluate_and_log, log_test_samples

def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASSES = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

    wandb.init(project="STL10-ResNet18", config={"epochs": args.epochs, "batch_size": 32})

    print("Loading data...")
    train_loader, test_loader = get_dataloaders(batch_size=32)

    print("Initializing model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Training on {DEVICE}...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            inputs = batch["pixel_values"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
        wandb.log({"train_loss": avg_loss})

    print("Evaluating Test Set...")
    acc = evaluate_and_log(model, test_loader, DEVICE, CLASSES)
    wandb.log({"final_accuracy": acc})
    print(f"Test Accuracy: {acc:.4f}")

    print("Logging 20 test samples to W&B...")
    log_test_samples(model, test_loader, DEVICE, CLASSES)

    # Save and Push to Hugging Face
    model_path = "resnet18_stl10.pt"
    torch.save(model.state_dict(), model_path)
    
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and args.repo_id != "YourUsername/STL10-ResNet18":
        print(f"Pushing to HF Hub: {args.repo_id}")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_path,
            repo_id=args.repo_id,
            repo_type="model",
        )
    else:
        print("Skipping HF push. Ensure HF_TOKEN is set and repo_id is updated.")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="YourUsername/STL10-ResNet18")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)