import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import wandb

def get_dataloaders(batch_size=32):
    # Load dataset
    dataset = load_dataset("Chiranjeev007/STL-10_Subset")
    
    # ResNet-18 normalization
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def transform_fn(examples):
        # FIX: Return ONLY the processed tensors and labels. 
        # This drops the raw PIL images that were crashing the DataLoader.
        return {
            "pixel_values": [preprocess(img.convert("RGB")) for img in examples["image"]],
            "label": examples["label"]
        }

    dataset.set_transform(transform_fn)
    
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size)
    
    return train_loader, test_loader

def evaluate_and_log(model, loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # 1. Log Visual Confusion Matrix to W&B
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=all_labels, preds=all_preds, class_names=class_names
        )
    })

    # 2. Log Class-wise Accuracy and Print for Exam Sheet
    cm = confusion_matrix(all_labels, all_preds)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print("\n" + "="*40)
    print("📋 EXAM SHEET REPORT (STEP 10)")
    print("="*40)
    
    for i, acc in enumerate(class_acc):
        wandb.log({f"class_accuracy/{class_names[i]}": acc})
        # This prints directly to your terminal so you can copy it to your sheet
        print(f"Class {i} ({class_names[i]}): {acc * 100:.2f}%")
        
    total_acc = np.sum(cm.diagonal()) / np.sum(cm)
    print(f"\nOverall Test Accuracy: {total_acc * 100:.2f}%")
    print("="*40 + "\n")
    
    return total_acc

def log_test_samples(model, test_loader, device, class_names):
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    # Un-normalize so images look normal on W&B instead of weird static
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(len(preds)):
                if len(correct_samples) >= 10 and len(incorrect_samples) >= 10:
                    break
                    
                pred_idx = preds[i].item()
                actual_idx = labels[i].item()
                
                # Format image for W&B
                img_tensor = inv_normalize(inputs[i]).cpu()
                img_tensor = torch.clamp(img_tensor, 0, 1)
                wandb_img = wandb.Image(img_tensor)
                
                row = [wandb_img, class_names[actual_idx], class_names[pred_idx]]
                
                if pred_idx == actual_idx and len(correct_samples) < 10:
                    correct_samples.append(row)
                elif pred_idx != actual_idx and len(incorrect_samples) < 10:
                    incorrect_samples.append(row)
                    
            if len(correct_samples) >= 10 and len(incorrect_samples) >= 10:
                break

    # Create W&B Table and log it
    table = wandb.Table(columns=["Image", "Actual Label", "Predicted Label"])
    for row in correct_samples + incorrect_samples:
        table.add_data(*row)
        
    wandb.log({"Test_Predictions_Sample": table})