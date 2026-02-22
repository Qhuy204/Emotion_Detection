import os
import torch
from typing import Dict, Any, Optional
from transformers import Trainer, TrainingArguments, IntervalStrategy
from training.losses import FocalLoss, get_phase_weights
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class EmotionTrainer(Trainer):
    """
    Custom Trainer that uses Focal Loss for better handling of class imbalance.
    """
    def __init__(
        self,
        *args,
        phase_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        alpha = phase_weights.tolist() if phase_weights is not None else None
        self.loss_fn = FocalLoss(
            alpha=alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is not None:
            loss = self.loss_fn(logits, labels)
        else:
            loss = outputs.loss

        # Put labels back for metrics
        inputs["labels"] = labels
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted")
    f1_macro = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro
    }


def get_training_args(config: Dict[str, Any], phase: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=os.path.join(config["training"]["output_dir"], phase),
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=config["data"]["batch_size"],
        per_device_eval_batch_size=config["data"]["eval_batch_size"],
        num_train_epochs=config["training"][f"num_epochs_{phase}"],
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=config["training"]["weight_decay"],
        logging_steps=config["training"]["logging_steps"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        save_total_limit=config["training"]["save_total_limit"],
        fp16=torch.cuda.is_available(),
        report_to="none",
        # New improvements
        warmup_ratio=config["training"].get("warmup_ratio", 0.1),
        lr_scheduler_type=config["training"].get("lr_scheduler_type", "cosine"),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )
