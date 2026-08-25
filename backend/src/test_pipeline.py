import torch
from transformers import AutoTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ToxicityClassifierDataset, InterventionGeneratorDataset
from src.models.classifier import ToxicityClassifier
from src.models.generator import InterventionGenerator

def test_pipeline():
    print("Testing PyTorch Pipeline Setup...\n")
    
    # 1. Test the Classifier Setup
    print("--- 1. Testing Classifier (DistilRoBERTa) ---")
    classifier_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    classifier_model = ToxicityClassifier()
    
    data_path = os.path.join("data", "processed", "classifier_dataset.csv")
    if os.path.exists(data_path):
        classifier_ds = ToxicityClassifierDataset(data_path, classifier_tokenizer, max_length=128)
        sample = classifier_ds[0]
        
        print(f"Dataset Size: {len(classifier_ds)}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Attention Mask shape: {sample['attention_mask'].shape}")
        print(f"Label: {sample['labels']}")
        
        # Test a dummy forward pass
        batch_inputs = sample['input_ids'].unsqueeze(0)
        batch_masks = sample['attention_mask'].unsqueeze(0)
        batch_labels = sample['labels'].unsqueeze(0)
        
        outputs = classifier_model(batch_inputs, batch_masks, batch_labels)
        print(f"Model Forward Pass successful. Loss: {outputs['loss'].item():.4f}\n")
    else:
        print(f"Warning: {data_path} not found. Run preprocessing first.\n")


    # 2. Test the Generator Setup
    print("--- 2. Testing Generator (FLAN-T5) ---")
    generator_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    generator_model = InterventionGenerator()
    
    gen_data_path = os.path.join("data", "processed", "generator_dataset.csv")
    if os.path.exists(gen_data_path):
        generator_ds = InterventionGeneratorDataset(gen_data_path, generator_tokenizer, max_source_length=128, max_target_length=48)
        gen_sample = generator_ds[0]
        
        print(f"Dataset Size: {len(generator_ds)}")
        print(f"Source Input IDs shape: {gen_sample['input_ids'].shape}")
        print(f"Source Attention Mask shape: {gen_sample['attention_mask'].shape}")
        print(f"Target Labels shape: {gen_sample['labels'].shape}")
        
        # Test a dummy forward pass
        batch_inputs = gen_sample['input_ids'].unsqueeze(0)
        batch_masks = gen_sample['attention_mask'].unsqueeze(0)
        batch_labels = gen_sample['labels'].unsqueeze(0)
        
        outputs = generator_model(batch_inputs, batch_masks, batch_labels)
        print(f"Model Forward Pass successful. Loss: {outputs.loss.item():.4f}\n")
        
    else:
        print(f"Warning: {gen_data_path} not found. Run preprocessing first.\n")
        
    print("Pipeline Verification Complete! All modules instantiated correctly.")

if __name__ == "__main__":
    test_pipeline()
