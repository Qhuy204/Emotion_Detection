import torch
import numpy as np
from transformers import AutoTokenizer
import argparse
import os


class EmotionPredictor:
    """Lightweight emotion predictor supporting both PyTorch and ONNX INT8 models."""

    ID2LABEL = {0: "positive", 1: "negative", 2: "surprise", 3: "neutral"}

    def __init__(self, model_path: str, use_onnx: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.use_onnx = use_onnx

        if use_onnx:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            self.model = ORTModelForSequenceClassification.from_pretrained(model_path)
        else:
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)

        if self.use_onnx:
            outputs = self.model(**inputs)
            pred = np.argmax(outputs.logits.numpy(), axis=-1).item()
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()

        return self.ID2LABEL[pred]


def export_and_quantize(model_path: str, onnx_path: str, int8_path: str) -> None:
    """Export PyTorch model to ONNX and then quantize to INT8."""
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    # Step 1: Export to ONNX
    print(f"[1/2] Exporting to ONNX: {model_path} → {onnx_path}")
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)
    ort_model.save_pretrained(onnx_path)

    # Copy tokenizer to ONNX dir
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(onnx_path)
    print(f"  ONNX model saved to {onnx_path}")

    # Step 2: Dynamic INT8 Quantization
    print(f"[2/2] Quantizing to INT8: {onnx_path} → {int8_path}")
    quantizer = ORTQuantizer.from_pretrained(onnx_path)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=int8_path, quantization_config=qconfig)

    # Copy all supporting files (config.json, tokenizer, etc.) to INT8 dir
    import shutil
    from pathlib import Path
    for f in Path(onnx_path).iterdir():
        if f.suffix != ".onnx" and f.name != "onnx":
            dest = Path(int8_path) / f.name
            if not dest.exists():
                if f.is_dir():
                    shutil.copytree(str(f), str(dest))
                else:
                    shutil.copy2(str(f), str(dest))
    print(f"  INT8 model saved to {int8_path}")

    # Show size comparison
    onnx_size = sum(f.stat().st_size for f in Path(onnx_path).rglob("*.onnx"))
    int8_size = sum(f.stat().st_size for f in Path(int8_path).rglob("*.onnx"))
    print(f"\n  Size comparison:")
    print(f"    ONNX FP32: {onnx_size / 1e6:.1f} MB")
    print(f"    ONNX INT8: {int8_size / 1e6:.1f} MB")
    if int8_size > 0:
        print(f"    Compression: {onnx_size / int8_size:.1f}x smaller")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Emotion / Export ONNX / Quantize INT8")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--text", type=str, help="Text to predict emotion for")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model for inference")
    parser.add_argument("--export", action="store_true", help="Export to ONNX + quantize INT8")
    parser.add_argument("--onnx_output", type=str, default="./outputs/onnx", help="ONNX output path")
    parser.add_argument("--int8_output", type=str, default="./outputs/int8", help="INT8 output path")

    args = parser.parse_args()

    if args.export:
        export_and_quantize(args.model_path, args.onnx_output, args.int8_output)
    elif args.text:
        predictor = EmotionPredictor(args.model_path, use_onnx=args.onnx)
        result = predictor.predict(args.text)
        print(f"Text: {args.text}")
        print(f"Emotion: {result}")
    else:
        print("Usage:")
        print("  Predict:  python predict.py --model_path ./outputs/int8 --onnx --text 'hello'")
        print("  Export:   python predict.py --model_path ./outputs/final_model --export")
