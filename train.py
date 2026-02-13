import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
from utils.preprocessing import get_transforms
import os
import json

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Dataset
# -----------------------------
DATA_DIR = "data/The IQ-OTHNCCD lung cancer dataset"
transform = get_transforms()

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Class mapping:", dataset.class_to_idx)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save class mapping
with open("models/class_mapping.json", "w") as f:
    json.dump(dataset.class_to_idx, f)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -----------------------------
# Model
# -----------------------------
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 30
best_val_acc = 0

train_losses = []
train_accuracies = []

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Compute metrics
    epoch_loss = running_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    train_losses.append(epoch_loss)
    train_accuracies.append(train_acc)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {epoch_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/lung_model.pth")
        print("✅ Best Model Saved")

print("🎉 Training Completed Successfully")

# -----------------------------
# Save Training History
# -----------------------------
history = {
    "loss": train_losses,
    "accuracy": train_accuracies
}

with open("training_history.json", "w") as f:
    json.dump(history, f)

print(" Training history saved successfully.")
