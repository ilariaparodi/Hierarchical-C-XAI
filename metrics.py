# metrics used for evaluation
import torch

# Accuracy: measures how well the model predicts the correct class label in a multi-class classification setting
def accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()

# Fine Concept Accuracy: measures how well the model predicts the correct fine concept labels in a multi-label classification setting
def fine_concept_accuracy(logits, targets, threshold=0.5):
    predictions = (torch.sigmoid(logits) >= threshold).float()
    return (predictions == targets).float().mean().item()

# Coarse Concept Accuracy: measures how well the model predicts the correct coarse concept labels in a multi-class classification setting
def coarse_concept_accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()

# Semantic Error Distance: measures the semantic distance between predicted and true labels in a hierarchical classification setting
def semantic_error_distance(predicted_coarse, predicted_class, true_coarse, true_class):

    total = 0
    n = len(true_class)

    for pc, pf, tc, tf in zip(predicted_coarse, predicted_class, true_coarse, true_class):

        if pf == tf:
            total += 0

        elif pc == tc:
            total += 1

        else:
            total += 2

    return total / n