import os
from datasets import load_dataset

subclass_map = {
    #Animal
    "feline": ["tabby cat", "tiger cat", "persian cat", "leopard", "tiger"],
    "bird": ["robin", "bulbul", "jay", "magpie", "bald eagle"],
    "fish": ["goldfish", "coho salmon", "anemone fish", "pufferfish"],

    #Vehicle
    "car": ["sport car", "station wagon", "racing car"],
    "truck": ["pickup truck", "tow truck", "trailer truck"],
    "bicycle": ["mountain bike", "tandem bicycle"],

    #Object
    "chair": ["folding chair", "barber chair"],
    "bottle": ["soda bottle", "water bottle", "wine bottle"],
    "clock": ["analog clock", "digital clock", "wall clock"],

    # Plants
    "flower": ["daisy", "lady's slipper"],
    "fruit": ["strawberry", "orange", "banana"]
}

def build_label_map(label_names):
    id_to_subclass = {}
    for id, name in enumerate(label_names):
        name_lower = name.lower()
        for subclass, keywords in subclass_map.items():
            for keyword in keywords:
                clean_keyword = keyword.lower().replace(" car", "").replace(" fish", "").replace("  truck", "")
                if clean_keyword in name_lower:
                    id_to_subclass[id] = subclass
                    break
    return id_to_subclass


def extract_dataset(n_images = 300):
    dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
    label_names = dataset.features["label"].names
    id_to_subclass = build_label_map(label_names)

    output_dir = "C:/Users/alege/Desktop/Io_quando_spiego/proj/image_subset"
    os.makedirs(output_dir, exist_ok=True)
    for subclass in subclass_map.keys():
        os.makedirs(os.path.join(output_dir, subclass), exist_ok=True)

    #1) class balancing operation:
    # compute exact targets per subclass to guarantee every category is present

    subclasses = list(subclass_map.keys())
    num_classes = len(subclasses)

    base_target = n_images // num_classes
    extra = n_images % num_classes

    target_counts = {}

    for i, subclass in enumerate(subclasses):
        # The first few classes get 1 extra image to handle the remainder and strictly hit 300
        target_counts[subclass] = base_target + (1 if i < extra else 0)

    current_counts = {subclass: 0 for subclass in subclasses}
    saved_count = 0

    for example in dataset:
        #2) Check if total target image is reached
        if saved_count >= n_images:
            break

        label_id = example['label']

        if label_id in id_to_subclass:
            subclass_name = id_to_subclass[label_id]

            if current_counts[subclass_name] >= target_counts[subclass_name]:
                continue

            image = example['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')

            filename = f"{subclass_name}_{current_counts[subclass_name]:03d}.jpg"
            filepath = os.path.join(output_dir, subclass_name, filename)
            image.save(filepath)

            current_counts[subclass_name] += 1
            saved_count += 1
            
            if saved_count % 20 == 0:
                print(f"Progress: Saved {saved_count}/{n_images} images...")

    print("\n--- Extraction Complete ---")
    for subclass, count in current_counts.items():
        print(f" -> {subclass}: {count} / {target_counts[subclass]}")

if __name__ == "__main__":
    extract_dataset(n_images=300)