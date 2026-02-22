# Emotion Detection

Multilingual emotion classification (English + Vietnamese) using transfer learning with XLM-RoBERTa → ONNX INT8 for lightweight deployment.

## Architecture

```
Input Text → XLM-RoBERTa-base (768 dim) → Classification Head → [positive, negative, surprise, neutral]
```

**Training Strategy**: Two-phase sequential transfer learning
- **Phase 1**: Pre-train on GoEmotions (211k English samples)
- **Phase 2**: Fine-tune on ViGoEmotions (16k Vietnamese samples)

**Key Techniques**: Class-weighted CE with label smoothing, cosine LR scheduler, warmup

## Project Structure

```
├── data/dataset.py              # DataModule (GoEmotions + ViGoEmotions)
├── models/
│   ├── architectures/classifier.py  # EmotionClassifier
│   └── configs/default.yaml         # Hyperparameters
├── training/
│   ├── losses.py                # BalancedCrossEntropyLoss with class weights
│   └── trainer.py               # Custom HF Trainer
├── train.py                     # Main training (Phase1 → Phase2 → ONNX → INT8)
├── predict.py                   # Inference + ONNX export + INT8 quantization
└── requirements.txt
```

## Usage

### Install
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py --config models/configs/default.yaml --hf_token HF_TOKEN
```
ONNX export runs automatically after training if `optimum` is installed.

### Inference (PyTorch)
```bash
python predict.py --model_path ./outputs/final_model --text "Cảm ơn bạn nhiều!"
```

### Inference (ONNX INT8 — recommended for production)
```bash
python predict.py --model_path ./outputs/int8 --onnx --text "Cảm ơn bạn nhiều!"
```

### Manual ONNX Export
```bash
python predict.py --model_path ./outputs/final_model --export
```

## Model Outputs

| Output | Path | Size | Use Case |
|:-------|:-----|:-----|:---------|
| Phase 1 (EN) | `outputs/phase1_final/` | ~1.1GB | Checkpoint |
| Final (VI) | `outputs/final_model/` | ~1.1GB | Full precision |
| ONNX FP32 | `outputs/onnx/` | ~1.1GB | ONNX runtime |
| **ONNX INT8** | **`outputs/int8/`** | **~280MB** | **Production / Chatbot** |

## Class Balancing

| Label | GoEmotions (EN) | ViGoEmotions (VI) |
|:------|:----------------|:------------------|
| positive | 76,245 | 6,505 |
| negative | 61,448 | 8,768 |
| surprise | 14,823 | 688 |
| neutral | 58,709 | 570 |

Handled via class-weighted CrossEntropyLoss (capped at 3.0) + label smoothing (0.1).
