
from pathlib import Path
import pickle
import faiss
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from facenet_pytorch import InceptionResnetV1
from insightface.app import FaceAnalysis

DB_PATH = Path("data/database.pkl")
INDEX_PATH = Path("data/index.faiss")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image: Image.Image):
    return transform(image).unsqueeze(0).to(device)

def build_model():
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return model

def build_model_arcface():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    return app

def build_model_vggface():
    # Use pretrained ResNet50 as an alternative to VGGFace
    base_model = models.resnet50(pretrained=True)
    # Modify the model to remove the classification head and use global average pooling
    model = nn.Sequential(
        *list(base_model.children())[:-1],  # Remove the final classification layer
        nn.AdaptiveAvgPool2d((1, 1)),      # Use adaptive average pooling
        nn.Flatten(),                      # Flatten the output
        nn.Linear(2048, 512),              # Map to 512-dimensional embeddings
        nn.ReLU(),                         # Applying ReLU activation
        nn.Linear(512, 512)                # Final embedding dimension
    ).to(device).eval()
    return model

def load_model():
    model = build_model()
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatIP(512)  # Must match embedding size

    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            database = pickle.load(f)
    else:
        database = []

    return model, index, database

def extract_embedding(model, image, model_name=None):
    if model_name == "arcface":
        np_image = np.asarray(image)
        faces = model.get(np_image)
        if not faces:
            return np.zeros((1, 512), dtype="float32")
        emb = faces[0].embedding
        emb = np.expand_dims(emb, axis=0)
    elif model_name == "vggface":
        img = image.resize((224, 224))  # Already set for ResNet50 input
        img_tensor = preprocess_image(img)

        with torch.no_grad():
            emb = model(img_tensor).cpu().numpy()
    else:
        x = preprocess_image(image)
        with torch.no_grad():
            emb = model(x).cpu().numpy()

    emb = normalize(emb, axis=1).astype("float32")
    return emb

def add_to_database(name, embedding, index, database, duplicate_threshold=0.95):
    embedding = normalize(embedding, axis=1).astype("float32")

    if index.ntotal > 0:
        D, I = index.search(embedding, 1)
        if D[0][0] > duplicate_threshold:
            print(f"Duplicate detected. Closest match: {database[I[0][0]][0]} with similarity {D[0][0]:.4f}")
            return False

    index.add(embedding)
    database.append((name, embedding))

    # Save
    with open(DB_PATH, "wb") as f:
        pickle.dump(database, f)

    faiss.write_index(index, str(INDEX_PATH))

    return True

def recognize(embedding, index, database, threshold=0.75):
    if index.ntotal == 0:
        return "No enrolled faces", None

    D, I = index.search(embedding, 1)
    if D[0][0] < threshold:
        return "Unknown", D[0][0]
    return database[I[0][0]][0], D[0][0]

