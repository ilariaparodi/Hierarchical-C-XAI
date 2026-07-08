import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from dataset.hierarchicalDataset import HierarchicalDataset # Your custom class

def compute_dataset_stats(loader):
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0

    # *_ multiple tensor approach
    for data, *_ in loader: 
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = torch.sqrt((channels_squared_sum / num_batches) - (mean ** 2))

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # basic tensor creation
    calc_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    #training data load
    train_dataset = HierarchicalDataset(
        annotation_file=os.path.join(BASE_DIR, "dataset", "train.json"),
        image_root=os.path.join(BASE_DIR, "image_subset"),
        transform=calc_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)


    #validation & test
    val_dataset = HierarchicalDataset(
        annotation_file=os.path.join(BASE_DIR, "dataset", "val.json"),
        image_root=os.path.join(BASE_DIR, "image_subset"),
        transform=calc_transform
    )
    
    test_dataset = HierarchicalDataset(
        annotation_file=os.path.join(BASE_DIR, "dataset", "test.json"),
        image_root=os.path.join(BASE_DIR, "image_subset"),
        transform=calc_transform
    )


    combined_val_test_dataset = ConcatDataset([val_dataset, test_dataset])
    val_test_loader = DataLoader(combined_val_test_dataset, batch_size=64, shuffle=False, num_workers=2)


    print("\n Computing Training Dataset stats.")
    train_mean, train_std = compute_dataset_stats(train_loader)

    print("\n Computing Combined Val/Test stats...")
    val_test_mean, val_test_std = compute_dataset_stats(val_test_loader)


    print("\n[ TRAINING DATASET ]")
    print(f"Mean: {train_mean}")
    print(f"Std:  {train_std}")

    print("\n[ VALIDATION + TESTING DATASETS ]")
    print(f"Mean: {val_test_mean}")
    print(f"Std:  {val_test_std}")