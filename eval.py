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
from sklearn.metrics.pairwise import cosine_distances
import kagglehub
from pathlib import Path
import faiss
from src import face_ops

# Load global config JSON
with open("configs/global_config.json", "r") as f:
    GLOBAL_CONFIG = json.load(f)

# Load model config JSON
with open("configs/model_config.json", "r") as f:
    MODEL_CONFIG = json.load(f)

# Download and locate dataset
DATASET_PATH = Path(kagglehub.dataset_download("vasukipatel/face-recognition-dataset"))
TEST_DIR = str(DATASET_PATH / "Original Images" / "Original Images")
print("Evaluating dataset at:", TEST_DIR)
LOAD_MODEL = GLOBAL_CONFIG.get("load_model", False)

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)


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

# List all image files in the directory
all_files = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(("jpg", "jpeg", "png"))
]

# Build list of (image_path, person_name) by parsing filename before the first underscore
# List all image files recursively, with folder name as person label
all_images = []

for person_dir in os.listdir(TEST_DIR):
    person_path = os.path.join(TEST_DIR, person_dir)
    if os.path.isdir(person_path):
        for img_file in os.listdir(person_path):
            if img_file.lower().endswith(("jpg", "jpeg", "png")):
                img_path = os.path.join(person_path, img_file)
                all_images.append((img_path, person_dir))

# Limit to first 500 images
all_images = all_images[:250]

# Rebuild list of unique people from the selected images
selected_people = set(person for _, person in all_images)

# Filter all_images again to keep only images of selected people (optional here)
test_set = [img for img in all_images if img[1] in selected_people]

print(f"Loaded {len(test_set)} test images.")

# Pre-enroll images for each person
enrollment_images = {}

for model_name, config in MODEL_CONFIG.items():
    print(f"\n--- Benchmarking {model_name} ---")

    builder_fn = getattr(face_ops, config["builder"])
    model = builder_fn()

    dim = config["dim"]
    metric = config["metric"].lower()

    # Only cosine similarity supported now, so fix index type
    index = faiss.IndexFlatIP(dim)  # Cosine similarity via inner product

    database = []

    finetune = GLOBAL_CONFIG.get("finetune", False)
    if isinstance(finetune, str):
        finetune = finetune.lower() == "true"

    # Initial enrollment: enroll multiple images per person for better cold start
    num_images_per_person = 3
    print(f"Initial enrollment with up to {num_images_per_person} images per person...")

    # Pre-enroll images for each person
    enrollment_images = {}

    # Group images by person
    for img_path, person in test_set:
        if person not in enrollment_images:
            enrollment_images[person] = []
        enrollment_images[person].append(img_path)

    # Select diverse images among all available for each person
    def select_diverse_images(image_paths, model, max_images=3):
        embeddings = []
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            embedding = face_ops.extract_embedding(model, image, model_name)  # Add correct model_name parameter
            if embedding is not None:
                embeddings.append((img_path, embedding))
        # Use all images if fewer than max_images
        if len(embeddings) <= max_images:
            return [e[0] for e in embeddings]

        # Calculate pairwise distances and choose diverse embeddings
        embeddings_array = np.vstack([e[1] for e in embeddings])
        diversity_scores = cosine_distances(embeddings_array, embeddings_array).sum(axis=1)
        selected = sorted(zip(diversity_scores, embeddings), key=lambda x: -x[0])[:max_images]
        return [s[1][0] for s in selected]  # Return selected image paths

    # Apply selection
    for person, image_paths in enrollment_images.items():
        diverse_paths = select_diverse_images(image_paths, model, max_images=3)
        enrollment_images[person] = diverse_paths

    print(f"Prepared enrollment images for {len(enrollment_images)} people with diversity.")
    
    if finetune:
        print(f"Fine-tuning enabled for {model_name}...")
        model = face_ops.finetune_model(model, enrollment_images, model_name, device)
        if model_name.lower() != "arcface":
           # Save the fine-tuned model with a timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(model.state_dict(), models_dir / f"{model_name}_finetuned_{timestamp}.pt")
    # Incremental evaluation and enrollment
    correct = 0
    total = 0
    similarities = []
    wrong_predictions = []

    # We iterate over the test set in order, simulating streaming input
    for img_path, true_label in tqdm(test_set):
        image = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            embedding = face_ops.extract_embedding(model, image, model_name)
        if embedding is None:
            print(f"ArcFace: No face detected in the image.")
            continue  # skip this image

        embedding = normalize(embedding, axis=1).astype("float32")
        predicted_name, sim = face_ops.recognize(embedding, index, database, threshold=0.75)

        total += 1
        if predicted_name == true_label:
            correct += 1
        else:
            wrong_predictions.append((img_path, true_label, predicted_name, float(sim)))

        similarities.append(float(sim))

        # Incrementally add this sample embedding to the database
        # You can restrict adding only if unknown or always add to improve coverage
        # Here we add always to simulate incremental enrollment
        face_ops.add_to_database(true_label, embedding, index, database)
        del embedding, image
        torch.cuda.empty_cache()

    accuracy = correct / total
    avg_sim = np.mean(similarities) if similarities else float("nan")

    print(f"Incremental Accuracy: {accuracy:.2%}")
    print(f"Average Similarity: {avg_sim:.4f}")

    # Visualization (same as before)
    plt.figure()
    plt.hist(similarities, bins=20, color="skyblue", edgecolor="black")
    plt.title(f"{model_name} Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)

    plot_path = os.path.join(results_subdir, f"{model_name}_similarity_distribution.png")
    plt.savefig(plot_path)

    if wrong_predictions:
        print("\nSample Wrong Predictions:")
        for path, true_label, pred, d in wrong_predictions[:5]:
            print(
                f"- {os.path.basename(path)} | True: {true_label}, Predicted: {pred}, Similarity: {d:.4f}"
            )

    results_summary[model_name] = {
        "accuracy": float(accuracy),
        "average_similarity": float(avg_sim),
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
    del model
    torch.cuda.empty_cache()

# Save all results with username and timestamp
user = getpass.getuser()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file = os.path.join(
    results_subdir, f"benchmark_results_{user}_{timestamp}.json"
)

with open(results_file, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\nBenchmarking complete. Results saved to {results_file}")
