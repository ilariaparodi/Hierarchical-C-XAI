# train a hierarchical concept bottleneck model
# with a ResNet50 backbone
import torch
from dataloaders import train_loader, val_loader, test_loader
from models.resnet_hierarchicalCBM import build_resnet50_hierarchical
from losses import total_loss
from metrics import accuracy, coarse_concept_accuracy, fine_concept_accuracy
import csv
import os

NUM_CLASSES = 4
NUM_COARSE = 11
NUM_FINE = 110

EPOCHS = 20
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "/content/drive/MyDrive/CBM_results"
os.makedirs(SAVE_DIR, exist_ok=True)

METRICS_FILE = os.path.join(SAVE_DIR, "hierarchical_metrics.csv")
BEST_MODEL = os.path.join(SAVE_DIR, "hierarchical_cbm_best.pt")

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
        lambda_class=1.0,    # final class prediction
        lambda_coarse=0.5,   # coarse concepts
        lambda_fine=0.1      # fine concepts
    )

    loss = losses["total"]
    return loss, losses, outputs, (final_class, coarse_concept, fine_concepts)

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

@torch.no_grad()

def evaluate(model, loader, device):
    
    model.eval()
    total = 0
    class_acc = 0.0
    coarse_acc = 0.0
    fine_acc = 0.0
    running_loss = 0.0
    
    # iterate over the validation data
    for batch in loader:
        loss, losses, outputs, targets = compute_loss(model, batch, device)
        final_class_targets, coarse_targets, fine_targets = targets
        
        bs = batch[0].size(0)

        # compute accuracies
        class_acc += accuracy(outputs["class"], final_class_targets) * bs
        coarse_acc += coarse_concept_accuracy(outputs["coarse"], coarse_targets) * bs
        fine_acc += fine_concept_accuracy(outputs["fine"], fine_targets) * bs

        running_loss += loss.item() * bs
        total += bs

    return {
        "loss": running_loss / total,
        "class_acc": class_acc / total,
        "coarse_acc": coarse_acc / total,
        "fine_acc": fine_acc / total,
    }

def main():
    print(f"Training Hierarchical Concept Bottleneck Model on {DEVICE}")

    with open(METRICS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "class_acc",
            "coarse_acc",
            "fine_acc",
        ])

    # build the model and optimizer
    model = build_resnet50_hierarchical(num_classes=NUM_CLASSES, num_fine=NUM_FINE, num_coarse=NUM_COARSE, pretrained=True, freeze_backbone=False).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_class_acc = 0.0 

    # train the model for a number of epochs
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Metrics -> Class Acc: {val_metrics['class_acc']:.4f} | Coarse Acc: {val_metrics['coarse_acc']:.4f} | Fine Acc: {val_metrics['fine_acc']:.4f}")

        with open(METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                val_metrics["loss"],
                val_metrics["class_acc"],
                val_metrics["coarse_acc"],
                val_metrics["fine_acc"],
            ])

        #based on final class accuracy 
        if val_metrics["class_acc"] > best_class_acc:
            best_class_acc = val_metrics["class_acc"]
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"*** New best model saved with Class Accuracy: {best_class_acc:.4f} ***")
        
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
        ])

    print("\n--- TEST ---")
    print(test_metrics)

if __name__ == "__main__":
    main()