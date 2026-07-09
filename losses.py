import torch
import torch.nn.functional as F

def classification_loss(predictions, labels):
    '''
    Cross entropy loss for the final classification task
    '''
    return F.cross_entropy(predictions, labels)


def concept_loss(coarse_predictions, coarse_targets, fine_predictions, fine_targets, lambda_coarse=0.3, lambda_fine=0.2):
    '''
    Hierarchical concept loss
    L_concept = λ_coarse * CE(coarse concepts) + λ_fine * BCE(fine concepts)
    '''
    coarse_loss = F.cross_entropy(coarse_predictions, coarse_targets )

    fine_loss = F.binary_cross_entropy_with_logits(fine_predictions, fine_targets.float())
    
    weighted_concept_loss = (lambda_coarse * coarse_loss) + (lambda_fine * fine_loss)

    return weighted_concept_loss, coarse_loss, fine_loss

def total_loss(class_predictions, class_targets, coarse_predictions, coarse_targets, fine_predictions, fine_targets,
                lambda_class=0.5, lambda_coarse=0.3, lambda_fine=0.2):
    '''
    Complete training loss
    L = lambda_class * L_classification + lambda_coarse * L_coarse_concept + lambda_fine * L_fine_concept
    '''
    l_class = classification_loss(class_predictions, class_targets)

    l_concept_weighted, l_coarse, l_fine = concept_loss(coarse_predictions, coarse_targets, fine_predictions, fine_targets, lambda_coarse, lambda_fine)

    l_total = (lambda_class * l_class) + l_concept_weighted

    return {
            "total": l_total,
            "class": l_class,
            "coarse": l_coarse,
            "fine": l_fine
        }