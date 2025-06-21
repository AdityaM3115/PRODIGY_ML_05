import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from food_dataset import FoodDataset

# Dataset paths
csv_path = "F:/vscode/python/task4/dataset/data.csv"
images_dir = "F:/vscode/python/task4/dataset/images"

# Log file for processed images and errors
log_file = "training_log.txt"
with open(log_file, "w") as f:
    f.write("Training Log\n")
    f.write("=================\n")

def log_message(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Image transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset and dataloader
dataset = FoodDataset(csv_file=csv_path, images_dir=images_dir, transform=data_transforms)

valid_data = []
for img, lbl in dataset:
    if img is not None and lbl is not None:
        valid_data.append((img, lbl))
        log_message(f"Processed: {lbl.item()}")
    else:
        log_message(f"Skipped file")

train_loader = DataLoader(valid_data, batch_size=32, shuffle=True)

# Load pretrained ResNet model
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Correct usage
model.fc = nn.Linear(model.fc.in_features, 1)  # Adjust final layer for kcal regression

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/food_calorie_model.pth")
print("Model saved successfully.")
