# train a hierarchical concept bottleneck model
# with a ResNet50 backbone
import torch
import matplotlib.pyplot as plt
import csv
import os
import random
import numpy as np
from dataset.dataloaders import train_loader, val_loader, test_loader
from models.resnet_hierarchicalCBM import build_resnet50_hierarchical
from losses import total_loss
from metrics import (
    accuracy,
    coarse_concept_accuracy,
    fine_concept_accuracy,

    fine_concept_precision_macro,
    fine_concept_recall_macro,
    fine_concept_f1_macro,

    fine_concept_precision_micro,
    fine_concept_recall_micro,
    fine_concept_f1_micro,
)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define constants and paths
NUM_CLASSES = 4
NUM_COARSE = 11
NUM_FINE = 110
EPOCHS = 20
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "/content/drive/MyDrive/CBM_results"
os.makedirs(SAVE_DIR, exist_ok=True)

METRICS_FILE = os.path.join(SAVE_DIR, "hierarchical_metrics.csv")
BEST_MODEL = os.path.join(SAVE_DIR, "hierarchical_cbm_best.pt")
LOSS_PLOT = os.path.join(SAVE_DIR, "hierarchical_loss.png")
ACC_PLOT = os.path.join(SAVE_DIR, "hierarchical_accuracy.png")

# Function to compute the loss for a given batch
def compute_loss(model, batch, device):
    # unpack the batch
    images, final_class, coarse_concept, fine_concepts = batch
    # move to device
    images = images.to(device)
    final_class = final_class.to(device)
    coarse_concept = coarse_concept.to(device)
    fine_concepts = fine_concepts.to(device)
    # forward pass
    outputs = model(images)
    # compute the total loss
    losses = total_loss(
        class_predictions=outputs["class"],
        class_targets=final_class,
        coarse_predictions=outputs["coarse"],
        coarse_targets=coarse_concept,
        fine_predictions=outputs["fine"],
        fine_targets=fine_concepts,
        lambda_class=0.4,    # final class prediction
        lambda_coarse=0.3,   # coarse concepts
        lambda_fine=0.3      # fine concepts
    )

    loss = losses["total"]
    return loss, losses, outputs, (final_class, coarse_concept, fine_concepts)

# Function to train the model for one epoch
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0
    n = 0

    # iterate over the training data
    for batch in loader:
        optimizer.zero_grad()
        loss, losses, _, _ = compute_loss(model, batch, device)
        loss.backward()
        optimizer.step()

        bs = batch[0].size(0)
        running_loss += loss.item() * bs
        n += bs

    return running_loss / n

# Function to evaluate the model
@torch.no_grad()
def evaluate(model, loader, device):
    
    model.eval()
    total = 0
    running_loss = 0.0
    class_acc = 0.0
    coarse_acc = 0.0
    fine_acc = 0.0

    # accumulate all predictions for macro/micro metrics
    all_logits = []
    all_targets = []

    for batch in loader:
        loss, losses, outputs, targets = compute_loss(model, batch, device)
        final_class_targets, coarse_targets, fine_targets = targets

        bs = batch[0].size(0)
        running_loss += loss.item() * bs
        total += bs

        class_acc += accuracy(outputs["class"], final_class_targets) * bs
        coarse_acc += coarse_concept_accuracy(outputs["coarse"], coarse_targets) * bs
        fine_acc += fine_concept_accuracy(outputs["fine"], fine_targets) * bs

        all_logits.append(outputs["fine"].detach().cpu())
        all_targets.append(fine_targets.detach().cpu())

    # concatenate predictions from all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # macro metrics
    macro_precision = fine_concept_precision_macro(all_logits, all_targets)
    macro_recall = fine_concept_recall_macro(all_logits, all_targets)
    macro_f1 = fine_concept_f1_macro(all_logits, all_targets)

    # micro metrics
    micro_precision = fine_concept_precision_micro(all_logits, all_targets)
    micro_recall = fine_concept_recall_micro(all_logits, all_targets)
    micro_f1 = fine_concept_f1_micro(all_logits, all_targets)

    return {
        "loss": running_loss / total,

        "class_acc": class_acc / total,
        "coarse_acc": coarse_acc / total,
        "fine_acc": fine_acc / total,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }

def main():
    print(f"Training Hierarchical CBM on {DEVICE}")

    with open(METRICS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "class_acc",
            "coarse_acc",
            "fine_acc",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "micro_precision",
            "micro_recall",
            "micro_f1"
        ])

    # build the model and optimizer
    model = build_resnet50_hierarchical(num_classes=NUM_CLASSES, num_fine=NUM_FINE, num_coarse=NUM_COARSE, pretrained=True, freeze_backbone=False).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # lists to track metrics across epochs for plotting
    train_losses = []
    val_losses = []
    class_accs = []
    coarse_accs = []
    fine_accs = []
    macro_precision = []
    macro_recall = []
    macro_f1 = []
    micro_precision = []
    micro_recall = []
    micro_f1 = []

    best_class_acc = 0.0 

    # train the model for a number of epochs
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        # save metrics for plotting and logging
        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        class_accs.append(val_metrics["class_acc"])
        coarse_accs.append(val_metrics["coarse_acc"])
        fine_accs.append(val_metrics["fine_acc"])
        macro_precision.append(val_metrics["macro_precision"])
        macro_recall.append(val_metrics["macro_recall"])
        macro_f1.append(val_metrics["macro_f1"])
        micro_precision.append(val_metrics["micro_precision"])
        micro_recall.append(val_metrics["micro_recall"])
        micro_f1.append(val_metrics["micro_f1"])

        print(f"\nEpoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(
            f"Metrics -> "
            f"Class Acc: {val_metrics['class_acc']:.4f} | "
            f"Coarse Acc: {val_metrics['coarse_acc']:.4f} | "
            f"Fine Acc: {val_metrics['fine_acc']:.4f}"
        )
        print(
            f"Macro -> "
            f"P: {val_metrics['macro_precision']:.4f} | "
            f"R: {val_metrics['macro_recall']:.4f} | "
            f"F1: {val_metrics['macro_f1']:.4f}"
        )
        print(
            f"Micro -> "
            f"P: {val_metrics['micro_precision']:.4f} | "
            f"R: {val_metrics['micro_recall']:.4f} | "
            f"F1: {val_metrics['micro_f1']:.4f}"
        )

        with open(METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                val_metrics["loss"],
                val_metrics["class_acc"],
                val_metrics["coarse_acc"],
                val_metrics["fine_acc"],

                val_metrics["macro_precision"],
                val_metrics["macro_recall"],
                val_metrics["macro_f1"],

                val_metrics["micro_precision"],
                val_metrics["micro_recall"],
                val_metrics["micro_f1"],
            ])

        # based on final class accuracy 
        if val_metrics["class_acc"] > best_class_acc:
            best_class_acc = val_metrics["class_acc"]
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"*** New best model saved with Class Accuracy: {best_class_acc:.4f} ***")

    # plot training and validation loss
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, label="train")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="val")
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(LOSS_PLOT, dpi=150, bbox_inches="tight")
    plt.close()

    # plot validation accuracies
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), class_accs, label="class")
    plt.plot(range(1, EPOCHS + 1), coarse_accs, label="coarse")
    plt.plot(range(1, EPOCHS + 1), fine_accs, label="fine")
    plt.title("validation accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(ACC_PLOT, dpi=150, bbox_inches="tight")
    plt.close()

    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, DEVICE)

    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "TEST",
            "",
            test_metrics["loss"],
            test_metrics["class_acc"],
            test_metrics["coarse_acc"],
            test_metrics["fine_acc"],

            test_metrics["macro_precision"],
            test_metrics["macro_recall"],
            test_metrics["macro_f1"],

            test_metrics["micro_precision"],
            test_metrics["micro_recall"],
            test_metrics["micro_f1"],
        ])

    print("\n--- TEST ---")
    print(test_metrics)

if __name__ == "__main__":
    main()
