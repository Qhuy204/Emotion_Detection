# Emotion Detection Project

This project refactors the original `emotional.ipynb` notebook into a modular, production-ready Python project. It implements a two-phase training strategy for cross-lingual emotion detection (English -> Vietnamese) with advanced class balancing.

## Project Structure

```text
.
├── data/
│   └── dataset.py          # Unified DataModule for GoEmotions and ViGoEmotions
├── models/
│   ├── architectures/
│   │   └── classifier.py   # EmotionClassifier architecture
│   └── configs/
│       └── default.yaml    # Hyperparameters and training config
├── training/
│   ├── losses.py           # Phase-specific weighted cross-entropy
│   └── trainer.py          # Custom Trainer with weighted loss
├── train.py                # Main training script (Phase 1 & Phase 2)
├── predict.py              # Inference and ONNX export script
└── requirements.txt        # Project dependencies
```

## How to use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training
Run the training script. It will automatically perform Phase 1 (pre-training on English) and Phase 2 (fine-tuning on Vietnamese).

```bash
python train.py --config models/configs/default.yaml --hf_token <YOUR_HF_TOKEN>
```
*Note: Ensure your HuggingFace token has access to the private `sonlam1102/vigoemotions` dataset.*

### 3. Inference
You can predict the emotion of a sentence using the trained model:

```bash
python predict.py --model_path ./outputs/final_model --text "Mình cảm thấy rất tuyệt!"
```

### 4. ONNX Export & Quantization
To export the final model to ONNX for optimized inference:

```bash
python predict.py --model_path ./outputs/final_model --export_onnx --onnx_output ./emotion_model_onnx
```

## Improvements over Notebook
1.  **Sequential Transfer Learning**: Pre-training on massive English data before fine-tuning on Vietnamese.
2.  **Class Weights**: Specific weights calculated for both datasets to solve the "Surprise" and "Neutral" (VI) class imbalance.
3.  **Modular Code**: Separated concerns make it easier to maintain and extend.
4.  **Config-Driven**: Easily experiment with learning rates, batch sizes, and hidden dimensions via YAML.
