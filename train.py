import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define image transformations (augmentation & normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit ResNet input size
    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation: Flip images
    transforms.RandomRotation(degrees=15),  # Data augmentation: Rotate images
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize values
])

# Load dataset from the dataset folder
dataset_path = "dataset"
train_dataset = ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform)
test_dataset = ImageFolder(root=os.path.join(dataset_path, "test"), transform=transform)

# Create DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print class names (should be ['damaged', 'non_damaged'])
print(f"Classes: {train_dataset.classes}")

# Load a pre-trained ResNet50 model
model = torchvision.models.resnet50(pretrained=True)

# Modify the last fully connected layer for binary classification (damaged vs non_damaged)
model.fc = nn.Linear(in_features=2048, out_features=2)
model = model.to(device)  # Move model to CPU/GPU

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Good for classification problems
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 10  # Adjust based on dataset size

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "models/damage_model.pth")
print("Model saved successfully!")

# Evaluate on test data
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
