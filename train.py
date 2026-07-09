import torch

from dataloaders import train_loader, val_loader, test_loader
from models.resnet_baseline import build_resnet50_baseline
from losses import classification_loss

NUM_CLASSES = 4
EPOCHS = 20
LR = 1e-3
TOP_K = 2
PRETRAINED = True
FREEZE_BACKBONE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def topk_correct(logits, targets, k):
    """Return (#top1_correct, #topk_correct) for a batch."""
    maxk = max(1, k)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()                                                # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))         # [maxk, B]

    top1 = correct[:1].reshape(-1).float().sum().item()
    topk = correct[:k].reshape(-1).float().sum().item()
    return top1, topk


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
def evaluate(model, loader, device, k=TOP_K):
    model.eval()
    total = 0
    top1_sum = 0.0
    topk_sum = 0.0
    loss_sum = 0.0

    for batch in loader:
        loss, logits, targets = compute_loss(model, batch, device)
        bs = targets.size(0)

        t1, tk = topk_correct(logits, targets, k)
        top1_sum += t1
        topk_sum += tk
        loss_sum += loss.item() * bs
        total += bs

    return {
        "loss": loss_sum / max(total, 1),
        "top1": top1_sum / max(total, 1),
        f"top{k}": topk_sum / max(total, 1),
    }


def main():
    print(f"Device: {DEVICE}")

    model = build_resnet50_baseline(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
    ).to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LR)

    best_val_top1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | "
            f"val_top1 {val_metrics['top1']:.3f} | "
            f"val_top{TOP_K} {val_metrics[f'top{TOP_K}']:.3f}"
        )

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            torch.save(model.state_dict(), "resnet50_baseline_best.pt")

    model.load_state_dict(torch.load("resnet50_baseline_best.pt", map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, DEVICE)
    print("\n--- TEST ---")
    print(
        f"test_top1 {test_metrics['top1']:.3f} | "
        f"test_top{TOP_K} {test_metrics[f'top{TOP_K}']:.3f} | "
        f"test_loss {test_metrics['loss']:.4f}"
    )


if __name__ == "__main__":
    main()
