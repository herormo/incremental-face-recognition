import os
import json
import torch
import matplotlib
import getpass
from datetime import datetime
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Fix PyCharm backend issue
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize
import kagglehub
from pathlib import Path
import faiss
from src import face_ops

# Load model config JSON
with open("configs/model_config.json", "r") as f:
    MODEL_CONFIG = json.load(f)

# Download and locate dataset
DATASET_PATH = Path(kagglehub.dataset_download("vasukipatel/face-recognition-dataset"))
TEST_DIR = str(DATASET_PATH / "Original Images" / "Original Images")
print("Evaluating dataset at:", TEST_DIR)

# Create results folder if it doesn't exist
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Create timestamped subdirectory
results_subdir = results_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_subdir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

results_summary = {}

# Preload all image paths and labels once
# only the last 100 people for testing
# Only load up to 100 images (across all people)
all_people_dirs = [
    (os.path.join(TEST_DIR, person), person)
    for person in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, person))
]

# Flatten all image paths with labels
all_images = []
for person_dir, person in all_people_dirs:
    for img_file in os.listdir(person_dir):
        if img_file.lower().endswith(("jpg", "jpeg", "png")):
            all_images.append((os.path.join(person_dir, img_file), person))
all_images = all_images[:100]

# Rebuild all_people_dirs to only include people in the selected 100 images
selected_people = set([person for _, person in all_images])
all_people_dirs = [
    (os.path.join(TEST_DIR, person), person) for person in selected_people
]

test_set = []
for person_dir, person in all_people_dirs:
    for img_file in os.listdir(person_dir):
        if img_file.lower().endswith(("jpg", "jpeg", "png")):
            img_path = os.path.join(person_dir, img_file)
            test_set.append((img_path, person))

print(f"Loaded {len(test_set)} test images.")

# Pre-enroll 3 images per person
enrollment_images = {}

for person_dir, person in all_people_dirs:
    for img_file in os.listdir(person_dir):
        if img_file.lower().endswith(("jpg", "jpeg", "png")):
            if person not in enrollment_images:
                enrollment_images[person] = []
            if len(enrollment_images[person]) < 2:  # Limit to 3 images
                enrollment_images[person].append(os.path.join(person_dir, img_file))

for model_name, config in MODEL_CONFIG.items():
    print(f"\n--- Benchmarking {model_name} ---")

    builder_fn = getattr(face_ops, config["builder"])
    model = builder_fn()
    dim = config["dim"]
    metric = config["metric"]

    if metric.lower() == "cosine":
        index = faiss.IndexFlatIP(dim)
    else:  # Default to L2 for non-normalized embeddings
        index = faiss.IndexFlatL2(dim)

    database = []
    # Enroll selected images per identity
    print("Enrolling samples per person...")
    for name, img_paths in enrollment_images.items():
        for img_path in img_paths:
            image = Image.open(img_path).convert("RGB")
            embedding = face_ops.extract_embedding(model, image, model_name)
            embedding = normalize(embedding, axis=1).astype("float32")
            face_ops.add_to_database(name, embedding, index, database)
    print(f"Enrolled {len(enrollment_images)} identities.")

    # Evaluate
    correct = 0
    total = len(test_set)
    similarities = []
    wrong_predictions = []

    for img_path, true_label in tqdm(test_set):
        image = Image.open(img_path).convert("RGB")
        embedding = face_ops.extract_embedding(model, image, model_name)
        predicted_name, dist = face_ops.recognize(embedding, index, database)

        if predicted_name == true_label:
            correct += 1
        else:
            wrong_predictions.append(
                (img_path, true_label, predicted_name, float(dist))
            )
        similarities.append(float(dist))

    accuracy = correct / total
    avg_dist = np.mean(similarities) if similarities else float("nan")

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Similarity (correct matches): {avg_dist:.4f}")

    # Visualization
    plt.figure()
    plt.hist(similarities, bins=20, color="skyblue", edgecolor="black")
    plt.title(f"{model_name} Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(
        results_subdir, f"{model_name}_similarity_distribution.png"
    )
    plt.savefig(plot_path)

    # Output error cases
    if wrong_predictions:
        print("\nSample Wrong Predictions:")
        for path, true_label, pred, d in wrong_predictions[:5]:
            print(
                f"- {os.path.basename(path)} | True: {true_label}, Predicted: {pred}, Similarity: {d:.4f}"
            )

    # Store results
    results_summary[model_name] = {
        "accuracy": float(accuracy),
        "average_similarity": float(avg_dist),
        "total": total,
        "correct": correct,
        "wrong_cases": [
            {
                "image": os.path.basename(p),
                "true": t,
                "pred": pr,
                "similarity": float(d),
            }
            for p, t, pr, d in wrong_predictions[:20]
        ],
    }

# Save all results with username and timestamp
user = getpass.getuser()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file = os.path.join(
    results_subdir, f"benchmark_results_{user}_{timestamp}.json"
)

with open(results_file, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\nBenchmarking complete. Results saved to {results_file}")
