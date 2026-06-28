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

        coarse = torch.tensor(sample["coarse_id"])
        fine = torch.tensor(sample["class_id"])

        concepts = torch.tensor(sample["concepts"],dtype=torch.float32)

        return image, coarse, fine, concepts
