import torch
from torch.utils.data import Dataset
import pandas as pd

class ToxicityClassifierDataset(Dataset):
    """Dataset for the Stage 1 Classifier model (e.g. DistilRoBERTa)."""
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        text = str(item['text'])
        label = int(item['label'])

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class InterventionGeneratorDataset(Dataset):
    """Dataset for the Stage 2 Generator model (e.g. FLAN-T5)."""
    def __init__(self, csv_file, tokenizer, max_source_length=128, max_target_length=48):
        self.data = pd.read_csv(csv_file)
        # Drop rows where toxic text or response is missing
        self.data = self.data.dropna(subset=['toxic_text', 'response'])
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        source_text = str(item['toxic_text'])
        target_text = str(item['response'])

        # Tokenize the input sequence
        source_encoding = self.tokenizer(
            source_text,
            truncation=True,
            max_length=self.max_source_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize the target sequence
        target_encoding = self.tokenizer(
            text_target=target_text,
            truncation=True,
            max_length=self.max_target_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels = target_encoding['input_ids'].flatten()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': labels
        }
