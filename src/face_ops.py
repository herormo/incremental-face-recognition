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
from facenet_pytorch import InceptionResnetV1, MTCNN
from insightface.app import FaceAnalysis


DB_PATH = Path("data/database.pkl")
INDEX_PATH = Path("data/index.faiss")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize MTCNN for face alignment
mtcnn = MTCNN(image_size=224, margin=30, thresholds=[0.6, 0.7, 0.7], device=device)

def preprocess_image(image: Image.Image):
    if isinstance(image, torch.Tensor):
        # If it's already a tensor
        if len(image.shape) == 3:  # Add batch dimension
            image = image.unsqueeze(0)
        return image.to(device)

    elif isinstance(image, Image.Image):  # Handle PIL image
        # Try face alignment with MTCNN
        if image.width > 1000 or image.height > 1000:
            image = image.resize((image.width // 2, image.height // 2))
        aligned = mtcnn(image)
        if aligned is None:
            print("Warning: No face detected by MTCNN. Falling back to raw resizing.")
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            return transform(image).unsqueeze(0).to(device)

        return aligned.unsqueeze(0).to(device)

    else:
        print("Error: Unsupported image format passed to preprocess_image.")
        return None

def build_model():
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model


def build_model_arcface():
    try:
        app = FaceAnalysis(name="buffalo_l")  # Use the correct ArcFace model name
        ctx_id = 0 if torch.cuda.is_available() else -1  # Use CUDA if available, fallback to CPU
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))  # Set detection size (optional, default is 640x640)
        return app
    except Exception as e:
        print(f"Error initializing ArcFace: {e}")
        raise


def build_model_resnet50():
    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Modify the model to remove the classification head and use global average pooling
    model = (
        nn.Sequential(
            *list(base_model.children())[:-1],  # Remove the final classification layer
            nn.AdaptiveAvgPool2d((1, 1)),  # Use adaptive average pooling
            nn.Flatten(),  # Flatten the output
            nn.Dropout(p=0.3),
            nn.Linear(2048, 512),  # Map to 512-dimensional embeddings
            nn.ReLU(),  # Applying ReLU activation
            nn.Linear(512, 512),  # Final embedding dimension
        )
        .to(device)
        .eval()
    )
    return model


def load_model():
    model = build_model_resnet50()
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
        np_image = np.asarray(image)
        print("Image shape:", np_image.shape, "dtype:", np_image.dtype)
        # ArcFace expects BGR (OpenCV format), convert if RGB
        if np_image.shape[2] == 3:
            np_image = np_image[:, :, ::-1].copy()  # RGB to BGR
        faces = model.get(np_image)
        if not faces:
            print("ArcFace: No face detected in the image.")
            return None

        embedding = faces[0].embedding
        embedding = np.expand_dims(embedding, axis=0)
        return normalize(embedding, axis=1).astype("float32")

    elif model_name in {"vggface2", "resnet50"}:
        image_tensor = preprocess_image(image)
        if image_tensor is None:
            print(f"Warning: Preprocessing failed for {model_name}.")
            return np.zeros((1, 512), dtype="float32")

        # Ensure image tensor has batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Fix: Ensure image has 3 channels
        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

        model.eval()
        with torch.no_grad():
            emb = model(image_tensor).cpu().numpy()

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Normalize embeddings for comparability
    emb = normalize(emb, axis=1).astype("float32")
    return emb



def add_to_database(name, new_embedding, index, database):
    new_embedding = normalize(new_embedding, axis=1).astype("float32")

       # Save all embeddings in the database
    database.append((name, new_embedding))
    index.add(new_embedding)
    with open(DB_PATH, "wb") as f:
        pickle.dump(database, f)
    faiss.write_index(index, str(INDEX_PATH))
    return True


def recognize(embedding, index, database, threshold=0.75):
    if index.ntotal == 0:
        return "No enrolled faces", 0.0  # Default similarity to 0.0

    D, indices = index.search(embedding, 1)  # Nearest neighbor search

    if D[0][0] >= threshold:  # Similarity meets/exceeds the threshold
        return database[indices[0][0]][0], D[0][0]
    else:
        return "Unknown", D[0][0]  # Return similarity even if below threshold

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
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)  # Remove extra batch dim if present
        return img_tensor, label

def finetune_model(model, enrollment_images, model_name, device, epochs=20, lr=1e-3, batch_size=16):
    classes = list(enrollment_images.keys())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    num_classes = len(classes)

    dataset = EnrollmentDataset(enrollment_images, class_to_idx, preprocess_image)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    # Special case for ArcFace: do not fine-tune the model, only train a classifier
    if model_name.lower() == "arcface":
        print("[ArcFace] Skipping fine-tuning; using ArcFace as frozen feature extractor.")
        return model 
    else:
        # Generic model: fine-tune the last layer or whole model
        model = model.to(device)
        model.train()

        # Add a classifier head if not already present
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
            parameters = model.parameters()
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes).to(device)
            parameters = model.parameters()
        else:
            # Add simple classifier on top of embeddings
            classifier = nn.Linear(512, num_classes).to(device)
            parameters = list(model.parameters()) + list(classifier.parameters())

        optimizer = torch.optim.Adam(parameters, lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for imgs, labels in dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)

                # If we're using a separate classifier
                if 'classifier' in locals():
                    outputs = classifier(outputs)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"[{model_name}] Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")

        model.eval()
        return model 
