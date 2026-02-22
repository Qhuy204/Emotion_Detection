import ast
from typing import Dict, List, Optional, Union
from datasets import load_dataset, DatasetDict, Dataset
import torch
from transformers import AutoTokenizer

class EmotionDataModule:
    """
    Data module for handling GoEmotions and ViGoEmotions datasets.
    Following ANTIGRAVITY standards for clarity and reproducibility.
    """
    
    CORE_LABELS = ["positive", "negative", "surprise", "neutral"]
    LABEL2ID = {k: v for v, k in enumerate(CORE_LABELS)}
    ID2LABEL = {i: l for l, i in LABEL2ID.items()}
    
    GO_MAP = {
        "joy": "positive", "love": "positive", "gratitude": "positive",
        "admiration": "positive", "approval": "positive", "optimism": "positive",
        "pride": "positive", "relief": "positive", "excitement": "positive",
        "amusement": "positive", "caring": "positive", "desire": "positive",
        "anger": "negative", "annoyance": "negative", "disappointment": "negative",
        "disapproval": "negative", "disgust": "negative", "embarrassment": "negative",
        "fear": "negative", "grief": "negative", "nervousness": "negative",
        "remorse": "negative", "sadness": "negative", "confusion": "negative",
        "surprise": "surprise", "curiosity": "surprise", "realization": "surprise",
        "neutral": "neutral"
    }
    
    VIGO_MAP = {
        **{i: "positive" for i in range(12)},
        **{i: "surprise" for i in range(12, 15)},
        **{i: "negative" for i in range(15, 27)},
        27: "neutral"
    }

    def __init__(self, model_name: str, max_length: int = 64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def _convert_go(self, example: Dict) -> Dict:
        mapped = []
        for emotion, target in self.GO_MAP.items():
            if example.get(emotion) == 1:
                mapped.append(target)
        
        if "negative" in mapped: final = "negative"
        elif "positive" in mapped: final = "positive"
        elif "surprise" in mapped: final = "surprise"
        else: final = "neutral"
        
        return {"text": example["text"], "label": self.LABEL2ID[final]}

    def _convert_vi(self, example: Dict) -> Dict:
        try:
            label_list = ast.literal_eval(example["labels"])
        except (ValueError, SyntaxError):
            label_list = []
            
        mapped = [self.VIGO_MAP.get(idx, "neutral") for idx in label_list]
        
        if "negative" in mapped: final = "negative"
        elif "positive" in mapped: final = "positive"
        elif "surprise" in mapped: final = "surprise"
        else: final = "neutral"
        
        return {"text": example["text"], "label": self.LABEL2ID[final]}

    def load_go_emotions(self) -> DatasetDict:
        dataset = load_dataset("mrm8488/goemotions")
        
        processed = DatasetDict()
        for split in dataset.keys():
            cols = dataset[split].column_names
            processed[split] = dataset[split].map(
                self._convert_go,
                remove_columns=cols,
                desc=f"Processing GoEmotions {split}"
            )
        return processed

    def load_vi_go_emotions(self, token: Optional[str] = None) -> DatasetDict:
        dataset = load_dataset("sonlam1102/vigoemotions", token=token)
        
        processed = DatasetDict()
        for split in dataset.keys():
            cols = dataset[split].column_names
            processed[split] = dataset[split].map(
                self._convert_vi,
                remove_columns=cols,
                desc=f"Processing ViGoEmotions {split}"
            )
        return processed

    def tokenize_function(self, examples: Dict) -> Dict:
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )

    def prepare_datasets(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        return dataset.map(self.tokenize_function, batched=True)
