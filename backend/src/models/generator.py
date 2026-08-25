import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class InterventionGenerator(nn.Module):
    """
    Wrapper around HuggingFace's Seq2Seq language model.
    Handles initialization of FLAN-T5.
    """
    def __init__(self, model_name="google/flan-t5-base"):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        # The underlying model handles the shift of labels internally for teacher forcing
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

    def generate(self, input_ids, attention_mask, max_new_tokens=48, **kwargs):
        """
        Helper method to generate text during inference.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
