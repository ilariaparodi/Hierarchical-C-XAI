import os
import json

# load concepts
with open("dataset_creation/concepts.json", "r") as f:
    concepts = json.load(f)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "image_subset")

# hierarchical mapping from subclasses to coarse classes
coarse_map = {
    # Animal
    "feline": "animal",
    "bird": "animal",
    "fish": "animal",

    # Vehicle
    "car": "vehicle",
    "truck": "vehicle",
    "bicycle": "vehicle",

    # Object
    "chair": "object",
    "bottle": "object",
    "clock": "object",

    # Plant
    "flower": "plant",
    "fruit": "plant"
}

# build a sorted list of all unique concepts across all subclasses
all_concepts = sorted(
    list(
        set(
            concept
            for subclass_concepts in concepts.values()
            for concept in subclass_concepts
        )
    )
)

print(f"Total concepts: {len(all_concepts)}")

with open("concept_vocabulary.json", "w") as f:
    json.dump(all_concepts, f, indent=4)

# build concept vectors for each subclass
concept_vectors = {}

for subclass, subclass_concepts in concepts.items():

    vector = []

    for concept in all_concepts:

        if concept in subclass_concepts:
            vector.append(1)
        else:
            vector.append(0)

    concept_vectors[subclass] = vector

# build mappings from class names to IDs
class_to_id = {
    subclass: idx
    for idx, subclass in enumerate(sorted(concepts.keys()))
}

coarse_to_id = {
    coarse: idx
    for idx, coarse in enumerate(
        sorted(set(coarse_map.values()))
    )
}

# create annotations for each image in the dataset
annotations = []

for coarse_class in os.listdir(DATASET_ROOT):

    coarse_dir = os.path.join(
        DATASET_ROOT,
        coarse_class
    )

    if not os.path.isdir(coarse_dir):
        continue

    for subclass in os.listdir(coarse_dir):

        subclass_dir = os.path.join(
            coarse_dir,
            subclass
        )

        if not os.path.isdir(subclass_dir):
            continue

        if subclass not in concept_vectors:
            continue

        for filename in os.listdir(subclass_dir):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            relative_path = os.path.join(
                coarse_class,
                subclass,
                filename
            )

            annotations.append({

                "image": relative_path,

                "coarse_class": coarse_class,
                "coarse_id": coarse_to_id[coarse_class],

                "class_name": subclass,
                "class_id": class_to_id[subclass],

                "active_concepts": concepts[subclass],
                "concepts": concept_vectors[subclass]
            })

# save annotations to JSON file
with open("annotations.json", "w") as f:
    json.dump(annotations, f, indent=4)

print(f"\nSaved {len(annotations)} image annotations")
print("Saved concept_vocabulary.json")
print("Saved annotations.json")
