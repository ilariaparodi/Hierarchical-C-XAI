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

# dictionaries
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

# load model
model = build_resnet50_hierarchical(num_classes=NUM_CLASSES, num_fine=NUM_FINE, num_coarse=NUM_COARSE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Function to display prediction results
def show_prediction(image, outputs, targets):
    # unpack the targets
    final_target, coarse_target, fine_target = targets
    # unpack the outputs
    pred_class = outputs["class"].argmax(1).item()
    pred_coarse = outputs["coarse"].argmax(1).item()
    fine_pred = (torch.sigmoid(outputs["fine"])>= 0.5).squeeze() 

    # get the predicted and ground truth fine concepts
    predicted_fine_concepts = [ concept_names[i] for i in range(len(concept_names)) if fine_pred[i] ]
    gt_fine_concepts = [ concept_names[i] for i in range(len(concept_names)) if fine_target.squeeze()[i] == 1]

    plt.figure(figsize=(5,5))
    img = image.squeeze().permute(1,2,0).cpu()
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print("="*60)
    print(
        f"Final class\n"
        f"GT   : {final_names[final_target.item()]}\n"
        f"Pred : {final_names[pred_class]}")
    print()
    print(
        f"Coarse concept\n"
        f"GT   : {coarse_names[coarse_target.item()]}\n"
        f"Pred : {coarse_names[pred_coarse]}")
    print()

    print("Ground Truth Fine Concepts")
    for c in gt_fine_concepts:
        print("  -", c)
    print()

    print("Predicted Fine Concepts")

    for c in predicted_fine_concepts:
        print("  -", c)
    print("="*60)

# Find correct and wrong examples
correct_found = False
wrong_found = False

# Iterate through the test dataset to find one correct and one wrong prediction
with torch.no_grad():
    for images, final_class, coarse, fine in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        pred_class = outputs["class"].argmax(1).cpu()

        for i in range(images.size(0)):
            # check if the prediction is correct
            if (not correct_found and pred_class[i] == final_class[i] ):
                print("\nCORRECT EXAMPLE\n")
                show_prediction(
                    images[i].cpu(),
                    {
                        "fine": outputs["fine"][i:i+1].cpu(),
                        "coarse": outputs["coarse"][i:i+1].cpu(),
                        "class": outputs["class"][i:i+1].cpu(),
                    },
                    (final_class[i], coarse[i], fine[i])
                )
                correct_found = True
            # check if the prediction is wrong
            if (not wrong_found and pred_class[i] != final_class[i] ):
                print("\nWRONG EXAMPLE\n")
                show_prediction(
                    images[i].cpu(),
                    {
                        "fine": outputs["fine"][i:i+1].cpu(),
                        "coarse": outputs["coarse"][i:i+1].cpu(),
                        "class": outputs["class"][i:i+1].cpu(),
                    },
                    (final_class[i], coarse[i], fine[i])
                )
                wrong_found = True

            if correct_found and wrong_found:
                raise SystemExit