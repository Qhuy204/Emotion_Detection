import torch
from transformers import AutoTokenizer
import argparse
import os
from optimum.onnxruntime import ORTModelForSequenceClassification

class EmotionPredictor:
    def __init__(self, model_path: str, use_onnx: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.id2label = {
            0: "positive",
            1: "negative",
            2: "surprise",
            3: "neutral"
        }
        
        if use_onnx:
            self.model = ORTModelForSequenceClassification.from_pretrained(model_path)
            self.is_onnx = True
        else:
            from models.architectures.classifier import EmotionClassifier, ModelConfig
            config = ModelConfig(model_name=model_path) # Simplified for loading
            # We can also use AutoModel
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            self.is_onnx = False

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt" if not self.is_onnx else "np",
            truncation=True,
            max_length=64,
            padding=True
        )
        
        if not self.is_onnx:
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
        else:
            # For ORTModel, inputs are handled internally
            import numpy as np
            inputs = {k: np.array(v) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            pred = np.argmax(logits, axis=-1).item()
            
        return self.id2label[pred]

def export_onnx(model_path: str, output_path: str):
    print(f"Exporting model from {model_path} to ONNX format...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        export=True
    )
    ort_model.save_pretrained(output_path)
    print(f"ONNX model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Emotion or Export to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--text", type=str, help="Text to predict emotion for")
    parser.add_argument("--export_onnx", action="store_true", help="Export model to ONNX")
    parser.add_argument("--onnx_output", type=str, default="./emotion_model_onnx", help="Path to save ONNX model")
    
    args = parser.parse_args()
    
    if args.export_onnx:
        export_onnx(args.model_path, args.onnx_output)
    elif args.text:
        predictor = EmotionPredictor(args.model_path)
        result = predictor.predict(args.text)
        print(f"Text: {args.text}")
        print(f"Predicted Emotion: {result}")
    else:
        print("Please provide --text for prediction or --export_onnx to export.")
