import argparse
import yaml
import os
import torch
from data.dataset import EmotionDataModule
from models.architectures.classifier import EmotionClassifier, ModelConfig
from training.trainer import EmotionTrainer, get_training_args, compute_metrics
from training.losses import get_phase_weights


def main():
    parser = argparse.ArgumentParser(description="Train Emotion Detection Model")
    parser.add_argument("--config", type=str, default="models/configs/default.yaml", help="Path to config file")
    parser.add_argument("--hf_token", type=str, help="HuggingFace Token for private datasets")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dm = EmotionDataModule(
        model_name=config["model"]["model_name"],
        max_length=config["data"]["max_length"]
    )

    id2label = EmotionDataModule.ID2LABEL
    label2id = EmotionDataModule.LABEL2ID

    model_config = ModelConfig(
        model_name=config["model"]["model_name"],
        num_labels=config["model"]["num_labels"],
        id2label=id2label,
        label2id=label2id,
        dropout=config["model"]["dropout"]
    )
    model = EmotionClassifier(model_config)

    label_smoothing = config["training"].get("label_smoothing", 0.1)

    # ── Phase 1: GoEmotions ──────────────────────────────────────
    print("=" * 60)
    print("  Phase 1: Pre-training on GoEmotions (English)")
    print("=" * 60)
    
    go_dataset = dm.load_go_emotions()
    if "test" not in go_dataset:
        go_dataset = go_dataset["train"].train_test_split(test_size=0.1, seed=42)
    go_dataset = dm.prepare_datasets(go_dataset)

    weights_p1 = torch.tensor(get_phase_weights("phase1"), dtype=torch.float)
    training_args_p1 = get_training_args(config, "phase1")

    trainer_p1 = EmotionTrainer(
        model=model.encoder,
        args=training_args_p1,
        train_dataset=go_dataset["train"],
        eval_dataset=go_dataset["test"],
        compute_metrics=compute_metrics,
        phase_weights=weights_p1,
        label_smoothing=label_smoothing,
    )

    trainer_p1.train()

    # Save Phase 1 model
    p1_path = os.path.join(config["training"]["output_dir"], "phase1_final")
    model.save_pretrained(p1_path)
    dm.tokenizer.save_pretrained(p1_path)
    print(f"Phase 1 model saved to {p1_path}")

    # ── Phase 2: ViGoEmotions ────────────────────────────────────
    print("=" * 60)
    print("  Phase 2: Fine-tuning on ViGoEmotions (Vietnamese)")
    print("=" * 60)

    # Resolve HF token: support both actual token and env var name
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if hf_token and not hf_token.startswith("hf_"):
        # User passed an env var name like "HF_TOKEN" instead of the actual token
        hf_token = os.getenv(hf_token, hf_token)

    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("Authenticated with HuggingFace Hub")
    else:
        print("WARNING: No HF_TOKEN provided. Gated datasets may not load.")

    vi_dataset = dm.load_vi_go_emotions(token=hf_token)
    vi_dataset = dm.prepare_datasets(vi_dataset)

    weights_p2 = torch.tensor(get_phase_weights("phase2"), dtype=torch.float)
    training_args_p2 = get_training_args(config, "phase2")

    trainer_p2 = EmotionTrainer(
        model=model.encoder,
        args=training_args_p2,
        train_dataset=vi_dataset["train"],
        eval_dataset=vi_dataset["validation"],
        compute_metrics=compute_metrics,
        phase_weights=weights_p2,
        label_smoothing=label_smoothing,
    )
    trainer_p2.train()

    # Save final model
    p2_path = os.path.join(config["training"]["output_dir"], "final_model")
    model.save_pretrained(p2_path)
    dm.tokenizer.save_pretrained(p2_path)
    print("=" * 60)
    print(f"  Training completed! Final model saved to {p2_path}")
    print("=" * 60)

    # ── ONNX Export + INT8 Quantization (optional) ────────────────
    try:
        from predict import export_and_quantize
        print("=" * 60)
        print("  Exporting to ONNX + INT8 Quantization")
        print("=" * 60)

        onnx_path = os.path.join(config["training"]["output_dir"], "onnx")
        int8_path = os.path.join(config["training"]["output_dir"], "int8")
        export_and_quantize(p2_path, onnx_path, int8_path)

        print("=" * 60)
        print(f"  All done! INT8 model ready at: {int8_path}")
        print("=" * 60)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"\n  Skipping ONNX export: {e}")
        print("  To enable, install: pip install optimum[onnxruntime]")
        print(f"  You can export later: python predict.py --model_path {p2_path} --export")


if __name__ == "__main__":
    main()
