import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    dataset_path = "/home/qhuy/Emotion_detection/Intent/intent_dataset.jsonl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run generate_dataset.py first.")
    
    # Load data
    texts = []
    labels_str = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels_str.append(obj["label"])
            
    # Label encoding
    unique_labels = sorted(list(set(labels_str)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    labels = [label2id[label] for label in labels_str]
    
    # Train-test split (80-20)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    # Create HuggingFace Datasets
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
    
    # Model configuration
    # Lựa chọn model nhẹ cho tiếng Việt (Khoảng 135M params)
    model_name = "vinai/phobert-base-v2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="/home/qhuy/Emotion_detection/Intent/results",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="/home/qhuy/Emotion_detection/Intent/logs",
        logging_steps=50,
        fp16=torch.cuda.is_available(), # Sử dụng mixed precision nếu có GPU
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    print("Bắt đầu training ...")
    trainer.train()
    
    # Save the best model
    output_model_dir = "/home/qhuy/Emotion_detection/Intent/phobert-intent-model"
    trainer.save_model(output_model_dir)
    print(f"Model đã được lưu tại {output_model_dir}")
    
if __name__ == "__main__":
    main()
