# split the dataset into train, validation, and test sets

import random
from collections import Counter, defaultdict
import json
import os

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS = os.path.join(BASE_DIR, "dataset_creation", "annotations.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Group samples by class name
groups = defaultdict(list)

with open(ANNOTATIONS) as f:
    annotations = json.load(f)

for sample in annotations:
    groups[sample["class_name"]].append(sample)

# Split each group into train, val, and test sets (70% train, 20% val, 10% test)
train = []
val = []
test = []

for samples in groups.values():
    random.shuffle(samples)
    n = len(samples)
    n_train = round(0.7 * n)
    n_val = round(0.2 * n)
    # n_test = n - n_train - n_val

    train.extend(samples[:n_train])
    val.extend(samples[n_train:n_train+n_val])
    test.extend(samples[n_train+n_val:])

# Save the splits to JSON files
with open(os.path.join(OUTPUT_DIR, "train.json"), "w") as f:
    json.dump(train, f, indent=4)

with open(os.path.join(OUTPUT_DIR, "val.json"), "w") as f:
    json.dump(val, f, indent=4)

with open(os.path.join(OUTPUT_DIR, "test.json"), "w") as f:
    json.dump(test, f, indent=4)

# Print the number of samples in each split
print(f"\nTrain images: {len(train)}")
print(f"Validation images: {len(val)}")
print(f"Test images: {len(test)}")