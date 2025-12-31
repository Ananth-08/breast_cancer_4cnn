import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# =====================
# CONFIG
# =====================
DATA_DIR = "data/400X"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/cnn_400x.pth"  # current saved model

print("Using device:", DEVICE)

# =====================
# TRANSFORMS
# =====================
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# DATASET
# =====================
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print("Classes:", test_ds.classes)

# =====================
# MODEL
# =====================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =====================
# EVALUATION
# =====================
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)

print(f"TEST Accuracy: {acc*100:.2f}%")
print(f"TEST Recall (Malignant): {rec*100:.2f}%")
print("Confusion Matrix:")
print(cm)
