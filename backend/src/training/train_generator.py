import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from src.data.dataset import InterventionGeneratorDataset
from src.models.generator import InterventionGenerator
import nltk
nltk.download('punkt_tab', quiet=True)

def train_generator(data_path="data/processed/generator_dataset.csv", max_steps=-1):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # 80% Train, 10% Val, 10% Test
    print("Splitting data (80/10/10)...")
    # For generation, all examples are toxic, so we don't necessarily stratify by label.
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    print("Initializing PyTorch Datasets...")
    train_dataset = InterventionGeneratorDataset(dataframe=train_df, tokenizer=tokenizer, max_source_length=128, max_target_length=48)
    val_dataset = InterventionGeneratorDataset(dataframe=val_df, tokenizer=tokenizer, max_source_length=128, max_target_length=48)
    test_dataset = InterventionGeneratorDataset(dataframe=test_df, tokenizer=tokenizer, max_source_length=128, max_target_length=48)

    model_wrapper = InterventionGenerator()

    # Load metrics
    metric_rouge = evaluate.load("rouge")
    metric_bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
            
        # Replace -100 in predictions and labels as we can't decode them
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Decode generated text and labels
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # ROUGE expects a newline after each sentence
        decoded_preds_rouge = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_rouge = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # Compute ROUGE
        rouge_output = metric_rouge.compute(predictions=decoded_preds_rouge, references=decoded_labels_rouge, use_stemmer=True)
        
        # Compute BLEU
        bleu_output = metric_bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
        
        return {
            "rouge1": rouge_output["rouge1"],
            "rouge2": rouge_output["rouge2"],
            "rougeL": rouge_output["rougeL"],
            "bleu": bleu_output["score"]
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results/generator",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        bf16=True, # bfloat16 mixed precision for speed (fixes NaN loss on T5)
        predict_with_generate=True, # Actually generate tokens to compute ROUGE
        generation_max_length=48,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        logging_steps=100,
        max_steps=max_steps # for sanity checking
    )

    trainer = Seq2SeqTrainer(
        model=model_wrapper.model, # Pass underlying HuggingFace model directly to Seq2SeqTrainer
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    # NLTK punkt needed for sentence tokenization in ROUGE
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    print("Starting training loop...")
    trainer.train()
    
    print("Evaluating on Test Set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")
    
    if max_steps == -1:
        # Only save if it's a full run
        trainer.save_model("./results/generator/best_model")
        tokenizer.save_pretrained("./results/generator/best_model")
        print("Model saved to ./results/generator/best_model")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="Run a 5-step training loop to test.")
    args = parser.parse_args()
    
    steps = 5 if args.sanity_check else -1
    train_generator(max_steps=steps)
