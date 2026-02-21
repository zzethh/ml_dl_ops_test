import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt # ADDED: For plotting

# ==========================
# Configuration
# ==========================
DATA_DIR = "data/train/"
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset & DataLoader
# ==========================
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes:", dataset.classes)

# ==========================
# Load ResNet-18
# ==========================
model = models.resnet18(pretrained=False)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)

# ==========================
# Loss & Optimizer
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================
# Training Loop
# ==========================
# ADDED: Lists to track metrics for plotting
train_losses = []
train_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    # ADDED: Save metrics for this epoch
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {epoch_loss:.4f} "
          f"Accuracy: {epoch_acc:.2f}%")

print("Training Complete!")

# ==========================
# Save Model
# ==========================
torch.save(model.state_dict(), "trained_model.pth")
print("Model saved!")

# ==========================
# Plot and Save Curves (ADDED FOR EXAM)
# ==========================
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, marker='o', color='red', label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, EPOCHS + 1))
plt.grid(True)
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_accuracies, marker='o', color='blue', label='Training Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(range(1, EPOCHS + 1))
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
print("Training curves successfully saved to 'training_curves.png'")