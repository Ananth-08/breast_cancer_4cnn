# 🩺 AI-Based Breast Cancer Detection System

An end-to-end deep learning application for breast cancer detection using histopathology images.
The system leverages multiple CNN models trained on different magnification levels and provides
a clean, interactive web interface for inference.

---

## 🔍 Project Overview

Breast cancer diagnosis from histopathology images is a complex and sensitive task that requires
high accuracy and reliability. This project implements a CNN-based image classification system
to predict whether a given histopathology image is benign or malignant.

The solution is designed as a professional-grade application with a clear separation between
model training and production deployment.

---

## 🧠 Key Features

- CNN-based image classification (Benign vs Malignant)
- Supports multiple magnification levels (40X, 100X, 200X, 400X)
- Multi-model inference using trained CNN weights
- FastAPI backend for real-time predictions
- Interactive React frontend
- Confidence score visualization
- Clean and maintainable project structure

---

## 📁 Project Structure

breast_cancer_4cnn/
│
├── training/                  # Model training & experiments (not deployed)
│   ├── modelstrained.ipynb
│   ├── split_data.py
│   ├── patient_split.csv
│   └── train_*.py
│
├── deployment/                # Final deployable application
│   ├── backend/
│   │   ├── api.py
│   │   ├── fusion_predict.py
│   │   ├── requirements.txt
│   │   └── models/
│   │       ├── cnn_40x.pth
│   │       ├── cnn_100x.pth
│   │       ├── cnn_200x.pth
│   │       └── cnn_400x.pth
│   │
│   └── frontend/
│       ├── package.json
│       ├── src/
│       └── public/
│
├── test_inputs/               # Sample images for testing
├── .gitignore
└── README.md

---

## ⚙️ Technology Stack

### Backend
- Python 3.10+
- PyTorch
- FastAPI
- Uvicorn

### Frontend
- React (Create React App)
- HTML / CSS
- JavaScript

---

## 🚀 Running the Application (Local)

### Backend Setup

cd deployment/backend  
pip install -r requirements.txt  
uvicorn api:app --reload  

Backend URL:  
http://127.0.0.1:8000  

API Documentation:  
http://127.0.0.1:8000/docs  

---

### Frontend Setup

cd deployment/frontend  
npm install  
npm start  

Frontend URL:  
http://localhost:3000  

---

## 🧪 Inference Workflow

1. Upload histopathology images (40X, 100X, 200X, 400X)
2. Click Predict
3. Backend performs CNN-based inference
4. Final prediction and confidence score are displayed

---

## 📌 Notes

- Training datasets, CSV split files, and notebooks are not part of deployment
- Only trained model weights are required for inference
- This project is intended for educational and research purposes

---

## 🔮 Future Enhancements

- Dockerized deployment
- Cloud hosting
- Batch inference support
- Model performance monitoring
- Automated medical report generation

---
