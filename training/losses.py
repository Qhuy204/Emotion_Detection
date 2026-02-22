from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with class weights and label smoothing.
    
    Simpler and more stable than Focal Loss for noisy multi-class tasks.
    Uses PyTorch's built-in label_smoothing for numerical stability.
    
    Args:
        weights: Per-class weights. Capped at max_weight to prevent instability.
        label_smoothing: Label smoothing factor. Default: 0.1.
        max_weight: Maximum allowed class weight. Default: 3.0.
    """
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        label_smoothing: float = 0.1,
        max_weight: float = 3.0,
    ):
        super().__init__()
        w = None
        if weights is not None:
            # Cap weights to prevent gradient instability
            capped = [min(w_i, max_weight) for w_i in weights]
            w = torch.tensor(capped, dtype=torch.float)
        self.register_buffer("weight", w)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, labels,
            weight=self.weight,
            label_smoothing=self.label_smoothing
        )


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
    min_w = min(weights)
    return [w / min_w for w in weights]
