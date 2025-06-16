from pathlib import Path
import pickle
import faiss
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
        return transform(image).to(device)  # Add batch and move to GPU/CPU

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
    model = build_model_vggface()
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
    if model_name == "arcface":
        # Convert PIL image to numpy array
        np_image = np.asarray(image)
        faces = model.get(np_image)
        if not faces:
            return np.zeros((1, 512), dtype="float32")
        emb = faces[0].embedding
        emb = np.expand_dims(emb, axis=0)

    elif model_name in {"facenet", "vggface"}:
        image_tensor = preprocess_image(image)
        if image_tensor is None:
            print(f"Warning: Preprocessing failed for {model_name}.")
            return np.zeros((1, 512), dtype="float32")

        # Ensure image tensor has batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            emb = model(image_tensor).cpu().numpy()

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Normalize embeddings for comparability
    emb = normalize(emb, axis=1).astype("float32")
    return emb



def add_to_database(name, embedding, index, database, duplicate_threshold=0.97):
    embedding = normalize(embedding, axis=1).astype("float32")

    # Remove old embeddings for this person
    database[:] = [item for item in database if item[0] != name]

    # Add the new embedding
    database.append((name, embedding))

    # Rebuild FAISS index from database embeddings
    all_embeddings = np.vstack([item[1] for item in database])
    index.reset()
    index.add(all_embeddings)

    # Save updated database and index
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

class EnrollmentDataset(Dataset):
    def __init__(self, enrollment_images, class_to_idx, preprocess_fn):
        self.samples = []
        self.labels = []
        for cls, paths in enrollment_images.items():
            for p in paths:
                self.samples.append(p)
                self.labels.append(class_to_idx[cls])
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess_fn(img)
        return img_tensor, label

def finetune_model(model, enrollment_images, model_name, device, epochs=20, lr=1e-4, batch_size=16):
    if model_name == "arcface":
        print("Finetuning not supported for ArcFace model.")
        return

    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    classes = list(enrollment_images.keys())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    dataset = EnrollmentDataset(enrollment_images, class_to_idx, preprocess_image)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            del imgs, labels, outputs, loss
            torch.cuda.empty_cache()

        avg_loss = running_loss / len(dataloader)
        print(f"Finetune Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()
