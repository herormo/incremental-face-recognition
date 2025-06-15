from PIL import Image
import torch
from src.face_ops import extract_embedding, recognize, load_model

# Path to image you want to test
IMAGE_PATH = "C:/Users/Alfy/Downloads/archive (1)/Original Images/Original Images/Zac Efron/Zac Efron_76.jpg"  # <- Replace this

# Load model, index, and database
model, index, database = load_model()

# Load and preprocess image
image = Image.open(IMAGE_PATH).convert("RGB")
embedding = extract_embedding(model, image)

# Perform recognition
identity, distance = recognize(embedding, index, database, threshold=1.2)

print(f"\nIdentified as: {identity}")
if distance is not None:
    print(f"Distance: {distance:.4f}")
