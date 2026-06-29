# define loss functions
# !!!
# scegliere valore di lambda
# valutare se lasciare nn.functional.binary_cross_entropy_with_logits o se usare nn.functional.binary_cross_entropy
# a seconda se il CBM produce logits o già probabilita' 
# !!!
import torch
import torch.nn.functional as F

def classification_loss(predictions, labels):
    '''
    Cross entropy loss.
    '''
    return F.cross_entropy(predictions, labels)


def concept_loss(coarse_predictions,coarse_targets,fine_predictions,fine_targets, lambda_weight=0.5):
    '''
    Hierarchical concept loss.
    L_concept = λ * BCE(coarse concepts) + (1-λ) * BCE(fine concepts)
    '''
    coarse_loss = F.binary_cross_entropy_with_logits(
        coarse_predictions,
        coarse_targets.float() )

    fine_loss = F.binary_cross_entropy_with_logits(
        fine_predictions,
        fine_targets.float() )

    return (lambda_weight * coarse_loss+ (1 - lambda_weight) * fine_loss)

def total_loss(class_predictions, class_targets, coarse_predictions, coarse_targets, fine_predictions, fine_targets, lambda_weight=0.5 ):
    '''
    Complete training loss.
    L = L_classification + L_concept
    '''

    l_class = classification_loss(class_predictions, class_targets)

    l_concept = concept_loss(coarse_predictions, coarse_targets, fine_predictions, fine_targets, lambda_weight)

    return l_class + l_concept
