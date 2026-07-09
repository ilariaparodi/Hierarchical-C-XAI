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
    return loss, losses, outputs, final_class

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
    running_loss = 0
    correct_class = 0
    correct_coarse = 0
    correct_fine = 0
    
    # iterate over the validation data
    for batch in loader:
        loss, losses, outputs, targets = compute_loss(model, batch, device)
        final_class_targets, coarse_targets, fine_targets = targets
        
        #final Class Accuracy (Argmax)
        pred_class = outputs["class"].argmax(dim=1)
        correct_class += (pred_class == final_class_targets).sum().item()
        
        #coarse Concept Accuracy (Argmax because of CrossEntropyLoss)
        pred_coarse = outputs["coarse"].argmax(dim=1)
        correct_coarse += (pred_coarse == coarse_targets).sum().item()
        
        #fine Concept Accuracy (Sigmoid + Threshold because of BCEWithLogitsLoss)
        pred_fine = (torch.sigmoid(outputs["fine"]) > 0.5).float()
        # Using .mean() gets the average element-wise match per batch, then we multiply by batch size
        correct_fine += (pred_fine == fine_targets).float().mean().item() * targets[0].size(0)

        running_loss += loss.item() * targets[0].size(0)
        total += targets[0].size(0)

    return {
        "loss": running_loss / max(1, total),
        "class_acc": correct_class / max(1, total),
        "coarse_acc": correct_coarse / max(1, total),
        "fine_acc": correct_fine / max(1, total)
    }

def main():
    print(f"Device: {DEVICE}")
    # build the model and optimizer
    model = build_resnet50_hierarchical(num_classes=NUM_CLASSES, num_fine=NUM_FINE, num_coarse=NUM_COARSE, pretrained=True, freeze_backbone=False).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_accuracy = -0.01   

    # train the model for a number of epochs
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Metrics -> Class Acc: {val_metrics['class_acc']:.4f} | Coarse Acc: {val_metrics['coarse_acc']:.4f} | Fine Acc: {val_metrics['fine_acc']:.4f}")

        #based on final class accuracy 
        if val_metrics["class_acc"] > best_accuracy:
            best_accuracy = val_metrics["class_acc"]
            torch.save(model.state_dict(), "hierarchical_cbm_best.pt")
            print(f"*** New best model saved with Class Accuracy: {best_accuracy:.4f} ***")
        
    model.load_state_dict(torch.load("hierarchical_cbm_best.pt", map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, DEVICE)

    print("\n--- TEST ---")
    print(test_metrics)

if __name__ == "__main__":
    main()