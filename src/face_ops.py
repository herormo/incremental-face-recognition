import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

from PIL import Image
import faiss
from pathlib import Path
import pickle
import numpy as np
from sklearn.preprocessing import normalize

DB_PATH = Path("data/database.pkl")
INDEX_PATH = Path("data/index.faiss")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
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
    model.fc = nn.Identity()  # use 512-d raw features
    model.to(device)
    model.eval()
    return model

def load_model():
    model = build_model()

    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatL2(512)  # must match embedding size

    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            database = pickle.load(f)
    else:
        database = []

    return model, index, database

def extract_embedding(model, image):
    x = preprocess_image(image)
    with torch.no_grad():
        emb = model(x).cpu().numpy()
    emb = normalize(emb, axis=1).astype("float32")
    return emb

def add_to_database(name, embedding, index, database):
    # Normalize again just to be safe
    embedding = normalize(embedding, axis=1).astype("float32")

    index.add(embedding)
    database.append((name, embedding))

    with open(DB_PATH, "wb") as f:
        pickle.dump(database, f)

    faiss.write_index(index, str(INDEX_PATH))

def recognize(embedding, index, database, threshold=1.85):
    embedding = normalize(embedding, axis=1).astype("float32")

    if index.ntotal == 0:
        return "No enrolled faces", None

    D, I = index.search(embedding, 1)

    if D[0][0] > threshold:
        return "Unknown", D[0][0]

    return database[I[0][0]][0], D[0][0]
