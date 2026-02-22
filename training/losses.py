from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with class weights to handle label imbalance.
    Weights should be provided as a list of floats.
    """
    def __init__(self, weights: Optional[List[float]] = None, device: str = "cpu"):
        super().__init__()
        self.weights = None
        if weights:
            self.weights = torch.tensor(weights, dtype=torch.float).to(device)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=self.weights)

def get_phase_weights(phase: str) -> List[float]:
    """
    Calculate class weights based on provided counts.
    Phase 1 (EN): {0: 76245, 1: 61448, 2: 14823, 3: 58709}
    Phase 2 (VI): {0: 6505, 1: 8768, 2: 688, 3: 570}
    
    Formula: Total / (num_classes * count)
    """
    if phase == "phase1":
        counts = [76245, 61448, 14823, 58709]
        total = sum(counts)
        return [total / (len(counts) * c) for c in counts]
    elif phase == "phase2":
        counts = [6505, 8768, 688, 570]
        total = sum(counts)
        return [total / (len(counts) * c) for c in counts]
    return [1.0, 1.0, 1.0, 1.0]
