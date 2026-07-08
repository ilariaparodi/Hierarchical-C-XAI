import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class HierarchicalDataset(Dataset):

    def __init__(self, annotation_file, image_root, transform=None):
        with open(annotation_file) as f:
            self.samples = json.load(f)

        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image"].replace('\\', '/')
        image = Image.open(os.path.join(self.image_root,image_path)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        final_class = torch.tensor(sample["final_class_id"], dtype=torch.long)
        coarse_concept = torch.tensor(sample["coarse_concept_id"], dtype=torch.long)
        fine_concepts = torch.tensor(sample["fine_concept_vector"], dtype=torch.float32)

        return image, final_class, coarse_concept, fine_concepts
