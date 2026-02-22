import torch
import unittest
from models.architectures.classifier import EmotionClassifier, ModelConfig

class TestModel(unittest.TestCase):
    def test_initialization(self):
        config = ModelConfig(num_labels=4)
        model = EmotionClassifier(config)
        self.assertEqual(model.config.num_labels, 4)
        
    def test_forward_pass(self):
        config = ModelConfig(num_labels=4)
        model = EmotionClassifier(config)
        
        # Mock inputs
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        self.assertEqual(outputs.logits.shape, (batch_size, 4))

if __name__ == "__main__":
    unittest.main()
