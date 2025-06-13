import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

import numpy as np
from PIL import Image
import faiss
from pathlib import Path
import pickle
import faiss

DB_PATH = Path("data/database.pkl")
INDEX_PATH = Path("data/index.faiss")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess_image(image: Image.Image):
    return transform(image).unsqueeze(0).to(device)

def build_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 128)
    return model.to(device)

def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatL2(512)

    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            database = pickle.load(f)
    else:
        database = []

    return model, index, database


def extract_embedding(model, image):
    model.to(device)  # move model to device

    x = preprocess_image(image)  # your preprocessing returns a tensor
    x = x.to(device)       # move input tensor to the same device

    with torch.no_grad():
        emb = model(x).cpu().numpy()  # output moved back to CPU numpy array

    return emb

def add_to_database(name, embedding, index, database):
    embedding_np = embedding.astype("float32")
    index.add(embedding_np)
    database.append((name, embedding_np))

    # Save
    with open(DB_PATH, "wb") as f:
        pickle.dump(database, f)
    faiss.write_index(index, str(INDEX_PATH))


def recognize(embedding, index, database, threshold=1.1):
    if index.ntotal == 0:
        return "No enrolled faces", None
    D, I = index.search(embedding, 1)
    if D[0][0] > threshold:
        return "Unknown", D[0][0]
    return database[I[0][0]][0], D[0][0]
