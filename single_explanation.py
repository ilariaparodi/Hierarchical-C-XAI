import json
import matplotlib.pyplot as plt
import torch

from dataset.dataloaders import test_loader
from models.resnet_hierarchicalCBM import build_resnet50_hierarchical

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 4
NUM_COARSE = 11
NUM_FINE = 110

MODEL_PATH = "/content/drive/MyDrive/CBM_results/hierarchical_cbm_best.pt"

# Change the path with the one of the image you want to inspect (among the test set)
IMAGE_PATH = "animal/bird/bird_023.jpg"

# Dictionaries

final_names = {
    0: "animal",
    1: "object",
    2: "plant",
    3: "vehicle"
}

coarse_names = {
    0: "bicycle",
    1: "bird",
    2: "bottle",
    3: "car",
    4: "chair",
    5: "clock",
    6: "feline",
    7: "fish",
    8: "flower",
    9: "fruit",
    10: "truck"
}

with open("dataset_creation/concept_vocabulary.json") as f:
    concept_names = json.load(f)

# Load model
model = build_resnet50_hierarchical(num_classes=NUM_CLASSES, num_fine=NUM_FINE, num_coarse=NUM_COARSE,)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# function to show the prediction of the model on a single image
def show_prediction(image, outputs, targets):

    final_target, coarse_target, fine_target = targets

    pred_class = outputs["class"].argmax(1).item()
    pred_coarse = outputs["coarse"].argmax(1).item()

    fine_pred = (torch.sigmoid(outputs["fine"]) >= 0.5).squeeze()

    gt_concepts = [
        concept_names[i]
        for i in range(len(concept_names))
        if fine_target[i] == 1
    ]

    pred_concepts = [
        concept_names[i]
        for i in range(len(concept_names))
        if fine_pred[i]
    ]

    # plot image
    plt.figure(figsize=(5,5))
    img = image.permute(1,2,0).cpu()
    # undo ImageNet normalization
    mean = torch.tensor([0.485,0.456,0.406])
    std = torch.tensor([0.229,0.224,0.225])
    img = img * std + mean
    img = img.clamp(0,1)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # print prediction results
    print("Final class")
    print(" GT   :", final_names[final_target.item()])
    print(" Pred :", final_names[pred_class])

    print()

    print("Coarse concept")
    print(" GT   :", coarse_names[coarse_target.item()])
    print(" Pred :", coarse_names[pred_coarse])

    print()

    print("Ground-truth fine concepts")
    for c in gt_concepts:
        print(" -", c)

    print()

    print("Predicted fine concepts")
    for c in pred_concepts:
        print(" -", c)

# Search image inside test set
found = False
dataset = test_loader.dataset

for idx in range(len(dataset)):
    sample = dataset.samples[idx]

    if sample["image"] == IMAGE_PATH:
        image, final_class, coarse, fine = dataset[idx]
        image_gpu = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image_gpu)
            
        show_prediction(
            image,
            {
                "class": outputs["class"].cpu(),
                "coarse": outputs["coarse"].cpu(),
                "fine": outputs["fine"].cpu(),
            },
            (
                final_class,
                coarse,
                fine,
            ),
        )

        found = True
        break

if not found:
    print(f"{IMAGE_PATH} not found in the test set.")