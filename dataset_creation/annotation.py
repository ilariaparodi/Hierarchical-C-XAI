import os
import json

with open("concepts.json", "w") as f:
    json.dump(concepts, f, indent=4)

DATASET_ROOT = "image_subset"

# build concept vocabulary
all_concepts = sorted(
    list(
        set(
            concept
            for subclass_concepts in concepts.values()
            for concept in subclass_concepts
        )
    )
)

# print(f"Total concepts: {len(all_concepts)}")

# save concept vocabulary
with open("concept_vocabulary.json", "w") as f:
    json.dump(all_concepts, f, indent=4)

# build concept vectors
concept_vectors = {}

for subclass, subclass_concepts in concepts.items():

    vector = []

    for concept in all_concepts:

        if concept in subclass_concepts:
            vector.append(1)
        else:
            vector.append(0)

    concept_vectors[subclass] = vector

# create image annotations
annotations = []

class_to_id = {
    subclass: idx
    for idx, subclass in enumerate(sorted(concepts.keys()))
}

for subclass in os.listdir(DATASET_ROOT):

    subclass_dir = os.path.join(DATASET_ROOT, subclass)

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
            subclass,
            filename
        )

        annotations.append({
            "image": relative_path,
            "class_name": subclass,
            "class_id": class_to_id[subclass],
            "concepts": concept_vectors[subclass]
        })

# save 
with open("annotations.json", "w") as f:
    json.dump(annotations, f, indent=4)

print(f"\nSaved {len(annotations)} image annotations")
print("Saved concept_vocabulary.json")
print("Saved annotations.json")
