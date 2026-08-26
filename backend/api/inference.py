import torch
import os
import sys
from transformers import AutoTokenizer

from safetensors.torch import load_file

# Ensure we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import ToxicityClassifier
from src.models.generator import InterventionGenerator

class InferencePipeline:
    def __init__(self, classifier_path, generator_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading models on {self.device} with FP16 precision...")

        # 1. Load Classifier Tokenizer and Model
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
        
        self.classifier = ToxicityClassifier()
        classifier_state_dict = load_file(os.path.join(classifier_path, "model.safetensors"))
        self.classifier.load_state_dict(classifier_state_dict, strict=False)
        self.classifier.half() # Convert to FP16
        self.classifier.to(self.device)
        self.classifier.eval()

        # 2. Load Generator Tokenizer and Model
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_path)
        
        self.generator = InterventionGenerator(model_name=generator_path)
        # The generator wrapper uses AutoModelForSeq2SeqLM.from_pretrained inside.
        # But wait, in train_generator.py we saved the model natively using trainer.save_model
        # So we can just re-instantiate with generator_path
        self.generator.half() # Convert to FP16
        self.generator.to(self.device)
        self.generator.eval()

        print("Models loaded successfully.")

    @torch.no_grad()
    def analyze(self, text: str) -> dict:
        # Classifier Inference
        inputs = self.classifier_tokenizer(
            text, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(self.device)

        outputs = self.classifier(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        toxic_prob = probs[0][1].item()
        
        is_toxic = toxic_prob > 0.5
        
        result = {
            "is_toxic": is_toxic,
            "confidence": toxic_prob if is_toxic else (1 - toxic_prob),
            "flagged_segment": text if is_toxic else None,
            "suggested_intervention": None
        }

        # If toxic, generate intervention
        if is_toxic:
            gen_prompt = f"Generate a constructive intervention for: {text}"
            gen_inputs = self.generator_tokenizer(
                gen_prompt, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(self.device)

            gen_outputs = self.generator.generate(
                input_ids=gen_inputs["input_ids"],
                attention_mask=gen_inputs["attention_mask"],
                max_new_tokens=48,
                num_beams=4,
                early_stopping=True
            )

            intervention = self.generator_tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
            result["suggested_intervention"] = intervention

        return result
