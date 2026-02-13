import torch
import json
import torch.nn.functional as F
from torchvision import models
from utils.preprocessing import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class mapping
with open("models/class_mapping.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_to_idx))
    model.load_state_dict(torch.load("models/lung_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(image):
    transform = get_transforms()
    image = transform(image).unsqueeze(0).to(device)

    model = load_model()

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = idx_to_class[predicted.item()]
    confidence_percent = confidence.item() * 100

    return predicted_class, round(confidence_percent, 2)
