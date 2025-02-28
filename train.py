import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from tqdm import tqdm  # Progress bar for training

# ✅ Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ✅ Define dataset paths
dataset_path = "dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# ✅ Validate dataset existence
if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("🚨 Dataset not found! Ensure 'train/' and 'test/' folders exist in 'dataset/'")

# ✅ Define image transformations (augmentations & normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
    transforms.RandomRotation(degrees=15),  # Data augmentation
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
])

# ✅ Load datasets
try:
    train_dataset = ImageFolder(root=train_path, transform=transform)
    test_dataset = ImageFolder(root=test_path, transform=transform)
except Exception as e:
    raise RuntimeError(f"🚨 Error loading dataset: {e}")

# ✅ Verify dataset has classes
num_classes = len(train_dataset.classes)
if num_classes == 0:
    raise ValueError("🚨 No valid classes found! Check dataset structure.")

print(f"📂 Detected {num_classes} classes: {train_dataset.classes}")

# ✅ Use `num_workers=0` for Windows to prevent crashes
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# ✅ Load a pre-trained ResNet50 model
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)  # Fixed deprecation warning

# ✅ Modify the last layer for multi-class classification
model.fc = nn.Linear(in_features=2048, out_features=num_classes)
model = model.to(device)  # Move model to GPU if available

# ✅ Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Multi-class classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Training parameters
num_epochs = 15
best_acc = 0.0  # Track best accuracy for saving the best model

# ✅ Wrap everything in `if __name__ == "__main__":` (Fix for Windows multiprocessing)
if __name__ == "__main__":
    print("\n🚀 Starting Training...\n")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # ✅ Progress bar
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            for images, labels in tepoch:
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

                tepoch.set_postfix(loss=running_loss / len(train_loader), accuracy=100 * correct / total)

        train_acc = 100 * correct / total
        print(f"✅ Epoch {epoch+1}: Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_acc:.2f}%")

        # ✅ Save the best model based on training accuracy
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), "models/best_damage_model.pth")
            print(f"📌 Best model updated (Accuracy: {best_acc:.2f}%)")

    # ✅ Save final trained model
    torch.save(model.state_dict(), "models/final_damage_model.pth")
    print("✅ Final model saved successfully!")

    # ✅ Evaluate on test data
    print("\n🔍 Evaluating on Test Data...\n")
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

    test_acc = 100 * correct / total
    print(f"🏆 Test Accuracy: {test_acc:.2f}%")

    # ✅ Save best model based on test accuracy
    if test_acc > best_acc:
        torch.save(model.state_dict(), "models/best_damage_model.pth")
        print(f"📌 Best model updated with Test Accuracy: {test_acc:.2f}%")

    print("\n✅ Training & Evaluation Completed Successfully! 🎉")
