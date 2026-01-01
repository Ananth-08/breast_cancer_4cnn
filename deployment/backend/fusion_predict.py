import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
THRESHOLD = 0.6   # medical-safe threshold

WEIGHTS = {
    "40X":  0.15,
    "100X": 0.25,
    "200X": 0.40,   # strongest
    "400X": 0.20
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# -------------------------------------------------
# IMAGE TRANSFORM (must match training)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# MODEL LOADER
# -------------------------------------------------
def load_model(weight_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

# -------------------------------------------------
# LOAD ALL 4 TRAINED MODELS
# -------------------------------------------------
model_40x  = load_model(os.path.join(MODEL_DIR,"cnn_40x.pth"))
model_100x = load_model(os.path.join(MODEL_DIR,"cnn_100x.pth"))
model_200x = load_model(os.path.join(MODEL_DIR,"cnn_200x.pth"))
model_400x = load_model(os.path.join(MODEL_DIR,"cnn_400x.pth"))

# -------------------------------------------------
# SINGLE MODEL PROBABILITY (MALIGNANT)
# -------------------------------------------------
def predict_prob(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img)
        prob = F.softmax(out, dim=1)

    return prob[0][1].item()  # malignant probability

# -------------------------------------------------
# FUSION PREDICTION
# -------------------------------------------------
def fusion_predict(img40, img100, img200, img400):
    p40  = predict_prob(model_40x, img40)
    p100 = predict_prob(model_100x, img100)
    p200 = predict_prob(model_200x, img200)
    p400 = predict_prob(model_400x, img400)

    final_prob = (
        WEIGHTS["40X"]  * p40 +
        WEIGHTS["100X"] * p100 +
        WEIGHTS["200X"] * p200 +
        WEIGHTS["400X"] * p400
    )

    label = "MALIGNANT" if final_prob >= THRESHOLD else "BENIGN"

    return {
        "prediction": label,
        "confidence_percent": round(final_prob * 100, 2),
        "individual_model_probs": {
            "40X": round(p40, 3),
            "100X": round(p100, 3),
            "200X": round(p200, 3),
            "400X": round(p400, 3)
        }
    }

# -------------------------------------------------
# MAIN: TEST INPUTS (MANUAL / BATCH)
# -------------------------------------------------
if __name__ == "__main__":

    BASE_DIR = "test_inputs"

    # --- Single patient example ---
    # result = fusion_predict(
    #     "test_inputs/patient01/40X.png",
    #     "test_inputs/patient01/100X.png",
    #     "test_inputs/patient01/200X.png",
    #     "test_inputs/patient01/400X.png"
    # )
    # print(result)

    # --- Batch testing (recommended) ---
    for patient_id in os.listdir(BASE_DIR):
        pdir = os.path.join(BASE_DIR, patient_id)
        if not os.path.isdir(pdir):
            continue

        result = fusion_predict(
            os.path.join(pdir, "40X.png"),
            os.path.join(pdir, "100X.png"),
            os.path.join(pdir, "200X.png"),
            os.path.join(pdir, "400X.png")
        )

        print(f"\nPatient: {patient_id}")
        print(result)
