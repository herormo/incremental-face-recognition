# Incremental Face Recognition System (PyTorch + FAISS + Streamlit)

A lightweight, modular face recognition system with incremental learning capabilities. Users can enroll new subjects and recognize faces using a simple web UI.

## Features

- Face embedding using ResNet18
- Fast similarity search using FAISS
- Streamlit-based user interface
- Modular PyTorch codebase
- Enroll and recognize faces without retraining

## Folder Structure

```text
incremental-face-recognition/
├── app.py              # Streamlit UI
├── src/
│   └── face_ops.py     # Embedding & FAISS logic
├── models/             # 
├── data/               # save/load index
├── requirements.txt
└── README.md
```

## Usage

- conda env create -f environment.yml
- conda activate face-recognition
- streamlit run app.py

## Todo

- [ ] Check uniqueness of enrolled face names to avoid duplicates
- [ ] In recognize mode, if a face is not recognized, automaticaly lead the user to change to enroll mode
- [ ] In enroll mode, check the database, if the face is recognized, suggest user to skip enrolling
- [ ] Evaluate and test accuracy (graphs) mechanism
- [ ] Use various models to benchmark
- [ ] Deploy online
- [ ] Improve/fix readme file
