from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Down-weights easy examples and focuses on hard/misclassified ones.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Per-class weights (list of floats). If None, all classes weighted equally.
        gamma: Focusing parameter. Higher γ = more focus on hard examples. Default: 2.0.
        label_smoothing: Label smoothing factor. Default: 0.0.
    """
    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        device: str = "cpu"
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha: Optional[torch.Tensor] = None
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float).to(device)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Move alpha to same device as logits if needed
        if self.alpha is not None and self.alpha.device != logits.device:
            self.alpha = self.alpha.to(logits.device)

        num_classes = logits.size(-1)
        
        # Apply label smoothing manually
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_labels = torch.zeros_like(logits)
                smooth_labels.fill_(self.label_smoothing / (num_classes - 1))
                smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # Compute softmax probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        if self.label_smoothing > 0:
            # Focal loss with label smoothing
            focal_weight = (1 - probs) ** self.gamma
            loss = -focal_weight * log_probs * smooth_labels
            
            if self.alpha is not None:
                alpha_weight = self.alpha.unsqueeze(0).expand_as(loss)
                loss = loss * alpha_weight
            
            loss = loss.sum(dim=-1).mean()
        else:
            # Standard focal loss
            ce_loss = F.nll_loss(log_probs, labels, reduction="none")
            p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.gamma
            loss = focal_weight * ce_loss
            
            if self.alpha is not None:
                alpha_t = self.alpha.gather(0, labels)
                loss = alpha_t * loss
            
            loss = loss.mean()

        return loss


def get_phase_weights(phase: str) -> List[float]:
    """
    Calculate class weights based on inverse frequency.
    
    Phase 1 (EN): {0: 76245, 1: 61448, 2: 14823, 3: 58709}
    Phase 2 (VI): {0: 6505, 1: 8768, 2: 688, 3: 570}
    
    Formula: Total / (num_classes * count), then normalized so min=1.0.
    """
    if phase == "phase1":
        counts = [76245, 61448, 14823, 58709]
    elif phase == "phase2":
        counts = [6505, 8768, 688, 570]
    else:
        return [1.0, 1.0, 1.0, 1.0]
    
    total = sum(counts)
    weights = [total / (len(counts) * c) for c in counts]
    # Normalize so minimum weight is 1.0
    min_w = min(weights)
    return [w / min_w for w in weights]
