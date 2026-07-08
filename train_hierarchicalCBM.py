# train a hierarchical concept bottleneck model
# with a ResNet50 backbone
import torch
from dataloaders import train_loader, val_loader, test_loader
from models.resnet_hierarchicalCBM import build_resnet50_hierarchical
from losses import total_loss

NUM_CLASSES = 4
NUM_COARSE = 11
NUM_FINE = 110

EPOCHS = 20
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    loss = total_loss(
        class_predictions=outputs["class"],
        class_targets=final_class,

        coarse_predictions=outputs["coarse"],
        coarse_targets=coarse_concept,

        fine_predictions=outputs["fine"],
        fine_targets=fine_concepts
    )

    return loss, outputs, final_class

def train_one_epoch(model, loader, optimizer, device):

    model.train()
    running_loss = 0
    n = 0

    # iterate over the training data
    for batch in loader:

        optimizer.zero_grad()

        loss, _, _ = compute_loss(model, batch, device)
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
    correct = 0
    running_loss = 0

    # iterate over the validation data
    for batch in loader:

        loss, outputs, labels = compute_loss(model, batch, device)
        pred = outputs["class"].argmax(dim=1)
        correct += (pred == labels).sum().item()
        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total
    }

def main():
    print(f"Device: {DEVICE}")
    # build the model and optimizer
    model = build_resnet50_hierarchical(pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_accuracy = 0.0

    # train the model for a number of epochs
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), "hierarchical_cbm_best.pt")
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
    model.load_state_dict(
    torch.load(
        "hierarchical_cbm_best.pt",
        map_location=DEVICE
    )
)

test_metrics = evaluate(model, test_loader, DEVICE)

print("\n--- TEST ---")
print(test_metrics)