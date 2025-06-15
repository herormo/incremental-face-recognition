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

def build_model_resnet18():
    return build_model()  # uses your existing resnet18 setup

def build_model_facenet():
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return model

def build_model_arcface():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    return app

def build_model_vggface():
    from keras_vggface.vggface import VGGFace
    from keras.models import Model
    base_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

def load_model():
    model = build_model()

    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatIP(512)  # must match embedding size

    if DB_PATH.exists():
        with open(DB_PATH, "rb") as f:
            database = pickle.load(f)
    else:
        database = []

    return model, index, database

def extract_embedding(model, image, model_name=None):
    if model_name == "arcface":
        # Convert PIL image to numpy array
        np_image = np.asarray(image)
        faces = model.get(np_image)
        if not faces:
            return np.zeros((1, 512), dtype="float32")
        emb = faces[0].embedding  # Access the attribute directly, not as dict
        emb = np.expand_dims(emb, axis=0)
    elif model_name == "vggface":
        from keras_vggface.utils import preprocess_input
        from tensorflow.keras.preprocessing import image as keras_image
        img = image.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array, version=2)
        emb = model.predict(img_array)
    else:
        x = preprocess_image(image)
        with torch.no_grad():
            emb = model(x).cpu().numpy()

    emb = normalize(emb, axis=1).astype("float32")
    return emb


def add_to_database(name, embedding, index, database, duplicate_threshold=0.2):
    # Normalize input embedding
    embedding = normalize(embedding, axis=1).astype("float32")

    # Check for duplicates
    if index.ntotal > 0:
        D, I = index.search(embedding, 1)
        if D[0][0] < duplicate_threshold:
            print(f"Duplicate detected. Closest match: {database[I[0][0]][0]} with distance {D[0][0]:.4f}")
            return False  # Skip adding

    # Add to index and database
    index.add(embedding)
    database.append((name, embedding))

    # Save
    with open(DB_PATH, "wb") as f:
        pickle.dump(database, f)
    faiss.write_index(index, str(INDEX_PATH))

    return True

def recognize(embedding, index, database, threshold=1.85):
    embedding = normalize(embedding, axis=1).astype("float32")

    if index.ntotal == 0:
        return "No enrolled faces", None

    D, I = index.search(embedding, 1)

    if D[0][0] < threshold:
        return "Unknown", D[0][0]

    return database[I[0][0]][0], D[0][0]
