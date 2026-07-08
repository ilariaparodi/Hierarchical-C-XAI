# create annotations for the image subset dataset
import os
import json

# define paths and load concepts
BASE_DIR = '/content/Hierarchical-C-XAI'
CONCEPTS_PATH = os.path.join(BASE_DIR, "dataset_creation", "concepts.json")

with open(CONCEPTS_PATH, "r") as f:
    concepts = json.load(f)

DATASET_ROOT = os.path.join(BASE_DIR, "image_subset")

# mapping from coarse semantic concepts to final classes
final_class_map = {
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

# build a sorted list of all unique fine concepts
all_fine_concepts = sorted(
    list(
        set(
            concept
            for fine_concepts in concepts.values()
            for concept in fine_concepts
        )
    )
)

print(f"Total fine concepts: {len(all_fine_concepts)}")

with open("concept_vocabulary.json", "w") as f:
    json.dump(all_fine_concepts, f, indent=4)

# build binary vectors for fine concepts
fine_concept_vectors = {}

for coarse_concept, fine_concepts in concepts.items():
    vector = []
    for concept in all_fine_concepts:
        vector.append(int(concept in fine_concepts))

    fine_concept_vectors[coarse_concept] = vector

# build mappings to integer IDs
coarse_concept_to_id = {
    concept: idx
    for idx, concept in enumerate(sorted(concepts.keys()))
}

final_class_to_id = {
    final_class: idx
    for idx, final_class in enumerate(
        sorted(set(final_class_map.values()))
    )
}

# create annotations for each image in the dataset
annotations = []

for final_class in os.listdir(DATASET_ROOT):
    final_class_dir = os.path.join(
        DATASET_ROOT,
        final_class
    )

    if not os.path.isdir(final_class_dir):
        continue

    for coarse_concept in os.listdir(final_class_dir):
        coarse_concept_dir = os.path.join(
            final_class_dir,
            coarse_concept
        )

        if not os.path.isdir(coarse_concept_dir):
            continue

        if coarse_concept not in fine_concept_vectors:
            continue

        for filename in os.listdir(coarse_concept_dir):
            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            relative_path = os.path.join(
                final_class,
                coarse_concept,
                filename
            )

            annotations.append({
                "image": relative_path,
                # final prediction target
                "final_class": final_class,
                "final_class_id": final_class_to_id[final_class],
                # intermediate semantic concept
                "coarse_concept": coarse_concept,
                "coarse_concept_id": coarse_concept_to_id[coarse_concept],
                # fine concepts
                "active_fine_concepts": concepts[coarse_concept],
                "fine_concept_vector": fine_concept_vectors[coarse_concept]
            })

# save annotations to JSON file
with open("annotations.json", "w") as f:
    json.dump(annotations, f, indent=4)

print(f"\nSaved {len(annotations)} image annotations")
print("Saved concept_vocabulary.json")
print("Saved annotations.json")
