import torch
import json
import os
import torch.nn.functional as F
from torchvision import models
from utils.preprocessing import get_transforms

# ==========================================
# DEVICE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# LOAD CLASS MAPPING
# ==========================================
with open("models/class_mapping.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# ==========================================
# LOAD MODEL (DenseNet121)
# ==========================================
def load_model():

    model_path = "models/lung_model.pth"   # ⭐ Recommended path

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n❌ Model not found at: {model_path}\n"
            "👉 Train model first OR move best_model.pth into models folder."
        )

    model = models.densenet121(weights=None)

    # SAME classifier used during training
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, len(class_to_idx))
    )

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    print("✅ Model loaded successfully")

    return model


# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_image(image, model):

    transform = get_transforms()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = idx_to_class[predicted.item()]
    confidence_percent = confidence.item() * 100

    return predicted_class, round(confidence_percent, 2)
