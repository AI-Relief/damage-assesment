import torch
import torchvision.models as models

# ✅ Define number of classes (same as used in training)
num_classes = 4  # ['Affected', 'Destroyed', 'Major', 'Minor']

# ✅ Load the model architecture (ResNet50)
model = models.resnet50()

# ✅ Modify the final fully connected (fc) layer to match num_classes
model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)

# ✅ Load the trained weights
model.load_state_dict(torch.load("models/best_damage_model.pth", map_location=torch.device('cpu')))

# ✅ Set to evaluation mode
model.eval()

# ✅ Print model summary
print("✅ Model Loaded Successfully!")
print(model)
