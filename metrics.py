# metrics used for evaluation
import torch

# top-1 accuracy: measures how often the top predicted label is the true label
def top1_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).float().mean().item()

# top-k accuracy: measures how often the true label is among the top k predicted labels
def topk_accuracy(logits, labels, k=5):
    _, topk = torch.topk(logits, k, dim=1)
    correct = topk.eq(labels.view(-1, 1))
    return correct.any(dim=1).float().mean().item()

# concept accuracy: measures how well the model predicts the presence of concepts in a multi-label classification setting
def concept_accuracy(logits, targets, threshold=0.5):
    """
    Computes binary concept accuracy.
    """
    predictions = torch.sigmoid(logits)
    predictions = (predictions >= threshold).float()
    correct = (predictions == targets).float()
    return correct.mean().item()

# misura se i coarse concept predetti sono coerenti con i fine concept
def hierarchical_consistency():
    """
    Computes hierarchical consistency between coarse and fine concepts.
    """
    pass  