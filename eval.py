
import os
import json
import torch
import matplotlib
matplotlib.use("Agg")  # Prevent GUI issues
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize
import faiss
from pathlib import Path

# Replace Keras-based VGGFace functionalities with PyTorch-based implementations
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms

# Helper function for preprocessing images
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# Function to build the ResNet50 model as a replacement for VGGFace
def build_model_vggface():
    base_model = models.resnet50(pretrained=True)
    model = nn.Sequential(
        *list(base_model.children())[:-1],  # Remove the classification layer
        nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        nn.Flatten(),  # Flatten the output
        nn.Linear(2048, 512),  # Projection to 512-dimensional embeddings
        nn.ReLU(),  # Activation
        nn.Linear(512, 512)  # Final embedding layer
    ).to(device).eval()
    return model

# Configuration for the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model configuration JSON
with open("configs/model_config.json", "r") as f:
    MODEL_CONFIG = json.load(f)

# Load dataset path
DATASET_PATH = Path("data/face-recognition-dataset")
TEST_DIR = str(DATASET_PATH / "Original Images" / "Original Images")

print("Evaluating dataset at:", TEST_DIR)

results_summary = {}

# Preload all image paths and labels once
all_people_dirs = [
    (os.path.join(TEST_DIR, person), person)
    for person in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, person))
]

test_set = []
for person_dir, person in all_people_dirs:
    for img_file in os.listdir(person_dir):
        if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(person_dir, img_file)
            test_set.append((img_path, person))

print(f"Loaded {len(test_set)} test images.")

# Pre-enroll one image per identity (for all models to use the same samples)
enrollment_images = {}
for person_dir, person in all_people_dirs:
    for img_file in os.listdir(person_dir):
        if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(person_dir, img_file)
            enrollment_images[person] = img_path
            break

# Iterate through each model in the configuration
for model_name, config in MODEL_CONFIG.items():
    print(f"\n--- Benchmarking {model_name} ---")

    # Dynamically call the model build function
    if model_name == "vggface":
        model = build_model_vggface()  # Use ResNet50 for the VGGFace replacement
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    dim = config["dim"]
    metric = config["metric"]

    if metric.lower() == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)

    database = []

    # Enroll selected images per identity
    print("Enrolling one sample per person...")
    for name, img_path in enrollment_images.items():
        image = Image.open(img_path).convert("RGB")
        embedding = preprocess_image(image)

        with torch.no_grad():
            embedding = model(embedding).cpu().numpy()

        embedding = normalize(embedding, axis=1).astype("float32")
        database.append((name, embedding))
        index.add(embedding)

    print(f"Enrolled {len(enrollment_images)} identities.")

    # Evaluate
    correct = 0
    total = len(test_set)
    distances = []
    wrong_predictions = []

    for img_path, true_label in tqdm(test_set):
        image = Image.open(img_path).convert("RGB")
        embedding = preprocess_image(image)

        with torch.no_grad():
            embedding = model(embedding).cpu().numpy()

        embedding = normalize(embedding, axis=1).astype("float32")
        dist, idx = index.search(embedding, 1)

        predicted_name = database[idx[0][0]][0]
        if predicted_name == true_label:
            correct += 1
            distances.append(float(dist))
        else:
            wrong_predictions.append((img_path, true_label, predicted_name, float(dist)))

    accuracy = correct / total
    avg_dist = np.mean(distances) if distances else float('nan')

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average distance (correct matches): {avg_dist:.4f}")

    # Visualize
    plt.figure()
    plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"{model_name} Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"{model_name}_similarity_distribution.png")

    # Output error cases
    if wrong_predictions:
        print("\nSample Wrong Predictions:")
        for path, true_label, pred, d in wrong_predictions[:5]:
            print(f"- {os.path.basename(path)} | True: {true_label}, Predicted: {pred}, Distance: {d:.4f}")

    # Store results
    results_summary[model_name] = {
        "accuracy": float(accuracy),
        "average_distance": float(avg_dist),
        "total": total,
        "correct": correct,
        "wrong_cases": [
            {
                "image": os.path.basename(p),
                "true": t,
                "pred": pr,
                "distance": float(d)
            } for p, t, pr, d in wrong_predictions[:20]
        ]
    }

# Save all results
with open("benchmark_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\nBenchmarking complete. Results saved to benchmark_results.json")

