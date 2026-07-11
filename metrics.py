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

# Fine Concept Recall: measures the proportion of true positive fine concept predictions out of all actual positive fine concept labels
def fine_concept_recall(logits, targets, threshold=0.5):
    predictions = (torch.sigmoid(logits) >= threshold).float()
    true_positives = ((predictions == 1) & (targets == 1)).float().sum(dim=0)
    actual_positives = targets.sum(dim=0)
    recall = true_positives / (actual_positives + 1e-8)
    return recall.mean().item()

# Fine Concept Precision: measures the proportion of true positive fine concept predictions out of all predicted positive fine concept labels
def fine_concept_precision(logits, targets, threshold=0.5):
    predictions = (torch.sigmoid(logits) >= threshold).float()
    true_positives = ((predictions == 1) & (targets == 1)).float().sum(dim=0)
    predicted_positives = predictions.sum(dim=0)
    precision = true_positives / (predicted_positives + 1e-8)
    return precision.mean().item()

# Fine Concept F1: measures the harmonic mean of precision and recall for fine concept predictions
def fine_concept_f1(logits, targets, threshold=0.5):
    recall = fine_concept_recall(logits, targets, threshold)
    precision = fine_concept_precision(logits, targets, threshold)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1