import os
import sys

from torchvision import transforms

from torch.utils.data import DataLoader

# Add the repository root to sys.path
repo_name = 'Hierarchical-C-XAI'
repo_root = os.path.join('/content', repo_name)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dataset.dataset import HierarchicalDataset

BASE_DIR = repo_root

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

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
