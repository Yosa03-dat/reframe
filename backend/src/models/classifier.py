import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class ToxicityClassifier(nn.Module):
    """
    Wrapper around HuggingFace's sequence classification model.
    Handles initialization of distilroberta and applies class weights to the loss function.
    """
    def __init__(self, model_name="distilroberta-base", num_labels=2, positive_weight=1.76):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # We manually define the loss function to account for our class imbalance
        # Weight for class 0 (non-toxic) = 1.0
        # Weight for class 1 (toxic) = 1.76
        # Register weights as a buffer so they automatically move to GPU with .to(device)
        self.register_buffer('class_weights', torch.tensor([1.0, positive_weight]))

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits
        loss = None
        
        if labels is not None:
            # Use registered buffer (already on the correct device)
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
