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

# Initialize MTCNN for face alignment
# mtcnn = MTCNN(image_size=224, margin=10, post_process=False, device=device)


def preprocess_image(image: Image.Image):
    if isinstance(image, torch.Tensor):
        # If it's already a tensor
        if len(image.shape) == 3:  # Add batch dimension
            image = image.unsqueeze(0)
        return image.to(device)

    elif isinstance(image, Image.Image):  # Handle PIL image
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform(image).unsqueeze(0).to(device)  # Add batch and move to GPU/CPU

    else:
        print("Error: Unsupported image format passed to preprocess_image.")
        return None


def build_model():
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model


def build_model_arcface():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    return app


def build_model_vggface():
    # Use pretrained ResNet50 as an alternative to VGGFace
    base_model = models.resnet50(pretrained=True)
    # Modify the model to remove the classification head and use global average pooling
    model = (
        nn.Sequential(
            *list(base_model.children())[:-1],  # Remove the final classification layer
            nn.AdaptiveAvgPool2d((1, 1)),  # Use adaptive average pooling
            nn.Flatten(),  # Flatten the output
            nn.Linear(2048, 512),  # Map to 512-dimensional embeddings
            nn.ReLU(),  # Applying ReLU activation
            nn.Linear(512, 512),  # Final embedding dimension
        )
        .to(device)
        .eval()
    )
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
    """
    Extract embeddings using different models based on the model_name.
    """
    # Preprocess the input image based on the model
    if model_name == "arcface":
        # Convert PIL image to numpy array
        np_image = np.asarray(image)
        faces = model.get(np_image)
        if not faces:
            return np.zeros((1, 512), dtype="float32")
        emb = faces[0].embedding  # Access the attribute directly, not as dict
        emb = np.expand_dims(emb, axis=0)

    elif model_name == "facenet":
        image_tensor = preprocess_image(image)  # Preprocess for Facenet
        if image_tensor is None:
            print("Warning: Preprocessing failed for Facenet.")
            return np.zeros((1, 512), dtype="float32")
        with torch.no_grad():
            emb = model(image_tensor).cpu().numpy()  # Get embeddings from Facenet

    elif model_name == "vggface":
        image_tensor = preprocess_image(image)  # Preprocess for ResNet50
        if image_tensor is None:
            print("Warning: Preprocessing failed for ResNet50.")
            return np.zeros((1, 512), dtype="float32")
        with torch.no_grad():
            emb = model(image_tensor).cpu().numpy()  # Get embeddings from ResNet50

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Normalize embeddings for comparability
    emb = normalize(emb, axis=1).astype("float32")
    return emb


def add_to_database(name, embedding, index, database, duplicate_threshold=0.97):
    embedding = normalize(embedding, axis=1).astype("float32")

    # Check for duplicates before adding
    # if index.ntotal > 0:
    #     D, I = index.search(embedding, 1)  # Search the closest match
    #     if D[0][0] > duplicate_threshold:  # Higher similarity means a duplicate
    #         print(f"Duplicate detected. Closest match: {database[I[0][0]][0]} with similarity {D[0][0]:.4f}")
    #         return False

    # Average embeddings if the identity already exists
    existing_embeddings = [data[1] for data in database if data[0] == name]
    if existing_embeddings:
        avg_embedding = np.mean(np.vstack([*existing_embeddings, embedding]), axis=0)
        embedding = normalize(avg_embedding.reshape(1, -1), axis=1).astype("float32")

    # Add embedding to FAISS and database
    index.add(embedding)
    database.append((name, embedding))

    # Save updated database and FAISS index
    with open(DB_PATH, "wb") as f:
        pickle.dump(database, f)
    faiss.write_index(index, str(INDEX_PATH))

    return True


def recognize(embedding, index, database, threshold=0.75):
    if index.ntotal == 0:
        return "No enrolled faces", None

    D, indices = index.search(embedding, 1)  # Nearest neighbor search

    # Analyze distances for matches
    if D[0][0] < threshold:  # If similarity is below the threshold
        return "Unknown", D[0][0]

    return database[indices[0][0]][0], D[0][0]
