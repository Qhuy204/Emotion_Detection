import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_intent(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get top prediction
    predicted_class_id = logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]
    confidence = probs[0][predicted_class_id].item()
    
    return predicted_label, confidence

def main():
    model_dir = "/home/qhuy/Emotion_detection/Intent/phobert-intent-model"
    
    if not os.path.exists(model_dir):
        print(f"Không tìm thấy model tại {model_dir}. Hãy chạy train_intent.py trước.")
        return
        
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    
    print("----- Intent Detection Tester -----")
    print("Gõ 'exit' hoặc 'quit' để thoát.")
    
    while True:
        text = input("\nBạn: ")
        if text.strip().lower() in ['exit', 'quit']:
            break
            
        if not text.strip():
            continue
            
        intent, conf = get_intent(text, model, tokenizer)
        print(f"Bot: Nhận diện Intent -> [{intent}] với độ chính xác {conf:.2%}")

if __name__ == "__main__":
    main()
