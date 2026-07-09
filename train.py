import torch
import matplotlib.pyplot as plt

from dataloaders import train_loader, val_loader, test_loader
from models.resnet_baseline import build_resnet50_baseline
from losses import classification_loss
from metrics import accuracy
import csv
import os

NUM_CLASSES = 4
EPOCHS = 20
LR = 1e-4
PRETRAINED = True
FREEZE_BACKBONE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "/content/drive/MyDrive/CBM_results"
os.makedirs(SAVE_DIR, exist_ok=True)

METRICS_FILE = os.path.join(SAVE_DIR, "baseline_metrics.csv")
BEST_MODEL = os.path.join(SAVE_DIR, "resnet50_baseline_best.pt")
LOSS_PLOT = os.path.join(SAVE_DIR, "baseline_loss.png")

def compute_loss(model, batch, device):
    images, final_class, coarse_concept, fine_concepts = batch
    images = images.to(device)
    final_class = final_class.to(device)

    logits = model(images)
    loss = classification_loss(logits, final_class)
    return loss, logits, final_class


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0
    n = 0
    for batch in loader:
        optimizer.zero_grad()
        loss, _, _ = compute_loss(model, batch, device)
        loss.backward()
        optimizer.step()

        bs = batch[0].size(0)
        running += loss.item() * bs
        n += bs
    return running / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    loss_sum = 0.0
    acc_sum = 0.0

    for batch in loader:
        loss, logits, targets = compute_loss(model, batch, device)
        bs = targets.size(0)

        acc = accuracy(logits, targets)
    
        loss_sum += loss.item() * bs
        acc_sum += acc * bs
        total += bs

    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": acc_sum / max(total, 1),
    }


def main():
    print(f"Device: {DEVICE}")

    with open(METRICS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
        ])

    model = build_resnet50_baseline(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
    ).to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LR)

    train_losses = []
    val_losses = []

    best_val_accuracy = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | "
            f"val_accuracy {val_metrics['accuracy']:.3f}"
        )

        with open(METRICS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["accuracy"],
            ])

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), BEST_MODEL)

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, label="train")
    plt.plot(range(1, EPOCHS + 1), val_losses, label="val")
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(LOSS_PLOT, dpi=150, bbox_inches="tight")
    plt.close()

    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, DEVICE)

    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "TEST",
            "",
            test_metrics["loss"],
            test_metrics["accuracy"],
        ])

    print("\n--- TEST ---")
    print(
        f"test_accuracy {test_metrics['accuracy']:.3f} | "
        f"test_loss {test_metrics['loss']:.4f}"
    )


if __name__ == "__main__":
    main()