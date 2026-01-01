from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os
from backend.fusion_predict import fusion_predict

app = FastAPI(title="Breast Cancer Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(
    img40: UploadFile = File(...),
    img100: UploadFile = File(...),
    img200: UploadFile = File(...),
    img400: UploadFile = File(...)
):
    os.makedirs("temp", exist_ok=True)

    files = {
        "40X.png": img40,
        "100X.png": img100,
        "200X.png": img200,
        "400X.png": img400
    }

    paths = {}
    for name, file in files.items():
        path = f"temp/{name}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        paths[name] = path

    return fusion_predict(
        paths["40X.png"],
        paths["100X.png"],
        paths["200X.png"],
        paths["400X.png"]
    )
