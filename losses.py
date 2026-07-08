import torch
import torch.nn.functional as F

def classification_loss(predictions, labels):
    '''
    Cross entropy loss for the final classification task
    '''
    return F.cross_entropy(predictions, labels)


def concept_loss(coarse_predictions, coarse_targets, fine_predictions, fine_targets, lambda_weight=0.6):
    '''
    Hierarchical concept loss
    L_concept = λ * CE(coarse concepts) + (1-λ) * BCE(fine concepts)
    '''
    coarse_loss = F.cross_entropy(
        coarse_predictions,
        coarse_targets )

    fine_loss = F.binary_cross_entropy_with_logits(
        fine_predictions,
        fine_targets.float() )

    return (lambda_weight * coarse_loss+ (1 - lambda_weight) * fine_loss)

def total_loss(class_predictions, class_targets, coarse_predictions, coarse_targets, fine_predictions, fine_targets, lambda_weight=0.6):
    '''
    Complete training loss
    L = L_classification + L_concept
    '''
    l_class = classification_loss(class_predictions, class_targets)

    l_concept = concept_loss(coarse_predictions, coarse_targets, fine_predictions, fine_targets, lambda_weight)

    return {
        "total": l_class + l_concept,
        "classification": l_class,
        "concept": l_concept
    }
