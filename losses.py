import torch
import torch.nn.functional as F

def classification_loss(predictions, labels):
    '''Cross entropy loss for the final classification task'''
    return F.cross_entropy(predictions, labels)


def concept_loss(coarse_predictions, coarse_targets,
                 fine_predictions, fine_targets,
                 lambda_coarse=0.3, lambda_fine=0.2,
                 pos_weight=None):                      # <-- NUOVO parametro
    '''
    Hierarchical concept loss
    L_concept = lambda_coarse * CE(coarse) + lambda_fine * BCE(fine)
    Se pos_weight e' fornito, la BCE sui fine e' pesata sui positivi.
    '''
    coarse_loss = F.cross_entropy(coarse_predictions, coarse_targets)

    # pos_weight: tensore (110,) che pesa gli errori sui concetti attivi
    fine_loss = F.binary_cross_entropy_with_logits(
        fine_predictions,
        fine_targets.float(),
        pos_weight=pos_weight                          # <-- passato qui
    )

    weighted_concept_loss = (lambda_coarse * coarse_loss) + (lambda_fine * fine_loss)
    return weighted_concept_loss, coarse_loss, fine_loss


def total_loss(class_predictions, class_targets,
               coarse_predictions, coarse_targets,
               fine_predictions, fine_targets,
               lambda_class=0.5, lambda_coarse=0.3, lambda_fine=0.2,
               pos_weight=None):                        # <-- NUOVO parametro
    '''
    Complete training loss
    L = lambda_class * L_class + lambda_coarse * L_coarse + lambda_fine * L_fine
    '''
    l_class = classification_loss(class_predictions, class_targets)

    l_concept_weighted, l_coarse, l_fine = concept_loss(
        coarse_predictions, coarse_targets,
        fine_predictions, fine_targets,
        lambda_coarse, lambda_fine,
        pos_weight=pos_weight                           # <-- propagato
    )

    l_total = (lambda_class * l_class) + l_concept_weighted

    return {
        "total": l_total,
        "class": l_class,
        "coarse": l_coarse,
        "fine": l_fine
    }