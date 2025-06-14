import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize
import kagglehub

from src.face_ops import extract_embedding, recognize, load_model, add_to_database

# Download and locate dataset
DATASET_PATH = kagglehub.dataset_download("vasukipatel/face-recognition-dataset")
TEST_DIR = str(DATASET_PATH / "face-recognition-dataset")  # Adjust if needed
print("Evaluating dataset at:", TEST_DIR)

THRESHOLD = 1.2

# Load model, index, and database
model, index, database = load_model()

# 1. Build test set
print("Building test set...")
test_set = []
for person in os.listdir(TEST_DIR):
    person_dir = os.path.join(TEST_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    for img_file in os.listdir(person_dir):
        if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
            img_path = os.path.join(person_dir, img_file)
            test_set.append((img_path, person))

print(f"Loaded {len(test_set)} test images.")

# 2. Evaluate
correct = 0
total = len(test_set)
distances = []
wrong_predictions = []

for img_path, true_label in tqdm(test_set):
    image = Image.open(img_path).convert("RGB")
    embedding = extract_embedding(model, image)
    predicted_name, dist = recognize(embedding, index, database, threshold=THRESHOLD)

    if predicted_name == true_label:
        correct += 1
        distances.append(dist)
    else:
        wrong_predictions.append((img_path, true_label, predicted_name, dist))

accuracy = correct / total
avg_dist = np.mean(distances) if distances else float('nan')

print(f"\nAccuracy: {accuracy:.2%}")
print(f"Average distance (correct matches): {avg_dist:.4f}")

# 3. Visualize
plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
plt.title("Distance Distribution for Correct Matches")
plt.xlabel("L2 Distance")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 4. Output error cases
if wrong_predictions:
    print("\nSample Wrong Predictions:")
    for path, true_label, pred, d in wrong_predictions[:5]:
        print(f"- {os.path.basename(path)} | True: {true_label}, Predicted: {pred}, Distance: {d:.4f}")

# 5. Export results
results = {
    "accuracy": accuracy,
    "average_distance": avg_dist,
    "total": total,
    "correct": correct,
    "wrong_cases": wrong_predictions[:20]  # limited for preview
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nEvaluation complete. Results saved to evaluation_results.json")
