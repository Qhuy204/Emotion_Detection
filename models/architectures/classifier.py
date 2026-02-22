from dataclasses import dataclass
from typing import Optional, Dict
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

@dataclass
class ModelConfig:
    """Configuration for Emotion Classifier."""
    model_name: str = "microsoft/Multilingual-MiniLM-L12-H384"
    num_labels: int = 4
    id2label: Optional[Dict[int, str]] = None
    label2id: Optional[Dict[str, int]] = None
    dropout: float = 0.1
    hidden_dim: int = 384

class EmotionClassifier(nn.Module):
    """
    Emotion Classifier based on pre-trained Transformer models.
    Following ANTIGRAVITY standards for performance and extensibility.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load base transformer model
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            id2label=config.id2label,
            label2id=config.label2id,
            classifier_dropout=config.dropout
        )
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        Returns SequenceClassifierOutput which includes loss and logits if labels are provided.
        """
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def save_pretrained(self, save_directory: str):
        """Save both model and its configuration."""
        self.encoder.save_pretrained(save_directory)
