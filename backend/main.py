
from fastapi import FastAPI, File, UploadFile
from PIL import Image

import io
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from pathlib import Path


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================================================
# Class names
# =========================================================

class_names = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip"
]


# =========================================================
# Load EfficientNet-B0
# =========================================================


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "efficientnet_b0_flowers_two.pth"
print("Model path:", MODEL_PATH)
print("Model exists:", MODEL_PATH.exists())
model = efficientnet_b0(weights=None)

# Change final classifier for 5 classes
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    5
)



# =========================================================
# Load trained model
# =========================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# =========================================================
# Image transformation
# =========================================================

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="Flower Classification API",
    description="EfficientNet-B0 Flower Classification Backend",
    version="1.0"
)


# =========================================================
# Home endpoint
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Flower Classification API is running!"
    }


# =========================================================
# Prediction endpoint
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Read uploaded image
    # -----------------------------------------------------

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    # -----------------------------------------------------
    # Transform image
    # -----------------------------------------------------

    image_tensor = transform(
        image
    )

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )


    # -----------------------------------------------------
    # Convert result
    # -----------------------------------------------------

    predicted_class = predicted_class.item()

    confidence = confidence.item()


    flower_name = class_names[
        predicted_class
    ]


    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return {
        "filename": file.filename,
        "prediction": flower_name,
        "confidence": round(
            confidence * 100,
            2
        )
    }