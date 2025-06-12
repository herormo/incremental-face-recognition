# Incremental Face Recognition System (PyTorch + FAISS + Streamlit)

A lightweight, modular face recognition system with incremental learning capabilities. Users can enroll new subjects and recognize faces using a simple web UI.

## Features
- Face embedding using ResNet18
- Fast similarity search using FAISS
- Streamlit-based user interface
- Modular PyTorch codebase
- Enroll and recognize faces without retraining

## Folder Structure
incremental-face-recognition/
├── app.py # Streamlit UI
├── src/
│ └── face_ops.py # Embedding & FAISS logic
├── models/ # (optional) save/load model
├── data/ # (optional) save/load index
├── requirements.txt
└── README.md

## Usage

- conda env create -f environment.yml
- conda activate face-recognition
- streamlit run app.py