import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os

# ✅ Define class names (Same order as in training)
class_names = ['Affected', 'Destroyed', 'Major', 'Minor']

# ✅ Load the trained model
num_classes = len(class_names)
model = models.resnet50()

# ✅ Modify the final layer to match number of classes
model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)

# ✅ Load the trained weights
model_path = "models/best_damage_model.pth"  # Change to "models/final_damage_model.pth" if needed
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

print(f"✅ Loaded model from {model_path}")

# ✅ Define image transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match ResNet input
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# ✅ Function to predict an image
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"🚨 Error: File {image_path} not found!")
        return

    # Load image
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    print(f"🛠️ Predicted Class: {predicted_class}")

# ✅ Run Prediction (Change the image path as needed)
image_path = "dataset/test/Major/Major1.jpg"  # Change this to your image file
predict_image(image_path)
