import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# --- HARDCODED CONFIGURATION ---
MODEL_PATH = "resnet18_classifier_stable.pth"
IMAGE_SIZE = (128, 384)
CLASS_NAMES = ['pair', 'stack'] # Ensure this order matches your training

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define model architecture and load weights
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 3. Define minimal transforms and process image
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
except FileNotFoundError:
    print(f"Error: Image not found at {image_path}")
    sys.exit(1)

# 4. Predict and print result
with torch.no_grad():
    output = model(image_tensor)
    prob = torch.sigmoid(output).item()
    pred_idx = 1 if prob > 0.5 else 0
    confidence = prob if pred_idx == 1 else 1 - prob

print(f"Prediction: {CLASS_NAMES[pred_idx].upper()} ({confidence:.2%})")