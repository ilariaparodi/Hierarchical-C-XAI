# =====================================================================
# MODIFICHE a train_hierarchicalCBM.py per il Fix 2 (pos_weight)
# =====================================================================

# -------------------------------------------------------------------
# AGGIUNTA A) - funzione per calcolare pos_weight dal training set.
# Mettila vicino agli altri import / prima di main().
# -------------------------------------------------------------------
import torch
from dataset.dataloaders import train_loader   # gia' importato nel file

def compute_pos_weight(loader, num_fine=110, device="cpu"):
    """
    pos_weight[c] = (# immagini in cui il concetto c e' ASSENTE)
                    / (# immagini in cui il concetto c e' PRESENTE)
    Calcolato una sola volta sul training set.
    """
    pos_counts = torch.zeros(num_fine)   # quante volte ogni concetto e' attivo
    total = 0
    for _, _, _, fine in loader:          # (image, final_class, coarse, fine)
        pos_counts += fine.sum(dim=0)     # somma sui campioni del batch
        total += fine.size(0)
    neg_counts = total - pos_counts
    # clamp per evitare divisione per zero se un concetto non appare mai
    pos_weight = neg_counts / pos_counts.clamp(min=1.0)
    return pos_weight.to(device)


# -------------------------------------------------------------------
# AGGIUNTA B) - dentro main(), DOPO aver creato il modello,
# calcola il peso una volta e passalo a compute_loss.
# -------------------------------------------------------------------
# pos_weight = compute_pos_weight(train_loader, num_fine=NUM_FINE, device=DEVICE)
# print("pos_weight fine concepts -> min:", pos_weight.min().item(),
#       "max:", pos_weight.max().item(), "mean:", pos_weight.mean().item())


# -------------------------------------------------------------------
# AGGIUNTA C) - modifica compute_loss per accettare e propagare pos_weight.
# La firma diventa:  def compute_loss(model, batch, device, pos_weight=None):
# e nella chiamata a total_loss aggiungi  pos_weight=pos_weight
# -------------------------------------------------------------------
def compute_loss(model, batch, device, pos_weight=None):     # <-- +pos_weight
    images, final_class, coarse_concept, fine_concepts = batch
    images = images.to(device)
    final_class = final_class.to(device)
    coarse_concept = coarse_concept.to(device)
    fine_concepts = fine_concepts.to(device)

    outputs = model(images)

    losses = total_loss(
        class_predictions=outputs["class"],
        class_targets=final_class,
        coarse_predictions=outputs["coarse"],
        coarse_targets=coarse_concept,
        fine_predictions=outputs["fine"],
        fine_targets=fine_concepts,
        lambda_class=0.5,
        lambda_coarse=0.3,
        lambda_fine=0.2,
        pos_weight=pos_weight                                 # <-- passato qui
    )
    loss = losses["total"]
    return loss, losses, outputs, (final_class, coarse_concept, fine_concepts)

# -------------------------------------------------------------------
# E infine: train_one_epoch e evaluate devono passare pos_weight a compute_loss.
#   loss, losses, _, _ = compute_loss(model, batch, device, pos_weight)
# (in evaluate puoi passarlo o lasciarlo None: influisce solo sul valore
#  della loss riportata, non sulle metriche di accuracy/F1)
# -------------------------------------------------------------------
