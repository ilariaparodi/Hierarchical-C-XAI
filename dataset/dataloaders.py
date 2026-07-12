# create dataloaders for the hierarchical dataset
import os
import sys
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.hierarchicalDataset import HierarchicalDataset

BASE_DIR = '/content/Hierarchical-C-XAI'

def get_dataloaders(batch_size=32, num_workers=2):
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4675564467906952, 0.4438961446285248, 0.3896581530570984],
            std=[0.2790640592575073, 0.27465948462486267, 0.29026371240615845]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4675564467906952, 0.4438961446285248, 0.3896581530570984],
            std=[0.2790640592575073, 0.27465948462486267, 0.29026371240615845]
        )
    ])

    train_dataset = HierarchicalDataset(
            annotation_file=os.path.join(BASE_DIR, "dataset", "train.json"),
            image_root=os.path.join(BASE_DIR, "image_subset"),
            transform=train_transform
        )

    val_dataset = HierarchicalDataset(
        annotation_file=os.path.join(BASE_DIR, "dataset", "val.json"),
        image_root=os.path.join(BASE_DIR, "image_subset"),
        transform=val_test_transform
    )

    test_dataset = HierarchicalDataset(
        annotation_file=os.path.join(BASE_DIR, "dataset", "test.json"),
        image_root=os.path.join(BASE_DIR, "image_subset"),
        transform=val_test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = get_dataloaders()