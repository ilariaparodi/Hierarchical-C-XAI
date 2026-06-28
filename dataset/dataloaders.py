# create dataloaders for the hierarchical dataset
import os
import sys
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.hierarchicalDataset import HierarchicalDataset

BASE_DIR = '/content/Hierarchical-C-XAI'

# define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# create datasets for train, validation, and test splits
train_dataset = HierarchicalDataset(
    annotation_file=os.path.join(
        BASE_DIR,
        "dataset",
        "train.json"
    ),

    image_root=os.path.join(
        BASE_DIR,
        "image_subset"
    ),

    transform=transform
)

val_dataset = HierarchicalDataset(
    annotation_file=os.path.join(
        BASE_DIR,
        "dataset",
        "val.json"
    ),

    image_root=os.path.join(
        BASE_DIR,
        "image_subset"
    ),

    transform=transform
)

test_dataset = HierarchicalDataset(
    annotation_file=os.path.join(
        BASE_DIR,
        "dataset",
        "test.json"
    ),

    image_root=os.path.join(
        BASE_DIR,
        "image_subset"
    ),

    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)