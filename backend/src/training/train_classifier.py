import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from src.data.dataset import ToxicityClassifierDataset
from src.models.classifier import ToxicityClassifier

def compute_metrics(eval_pred):
    metric_f1 = evaluate.load("f1")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_acc = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    precision = metric_precision.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = metric_recall.compute(predictions=predictions, references=labels, average="macro")["recall"]
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]

    return {
        "macro_f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": acc
    }

def train_classifier(data_path="data/processed/classifier_dataset.csv", max_steps=-1):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Stratified Split: 80% Train, 10% Val, 10% Test
    print("Splitting data (80/10/10)...")
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    
    print("Initializing PyTorch Datasets...")
    train_dataset = ToxicityClassifierDataset(dataframe=train_df, tokenizer=tokenizer)
    val_dataset = ToxicityClassifierDataset(dataframe=val_df, tokenizer=tokenizer)
    test_dataset = ToxicityClassifierDataset(dataframe=test_df, tokenizer=tokenizer)

    model = ToxicityClassifier()

    training_args = TrainingArguments(
        output_dir="./results/classifier",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True, # Mixed precision for speed
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=100,
        max_steps=max_steps # for sanity checking
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print("Starting training loop...")
    trainer.train()
    
    print("Evaluating on Test Set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")
    
    if max_steps == -1:
        # Only save if it's a full run
        trainer.save_model("./results/classifier/best_model")
        tokenizer.save_pretrained("./results/classifier/best_model")
        print("Model saved to ./results/classifier/best_model")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check", action="store_true", help="Run a 5-step training loop to test.")
    args = parser.parse_args()
    
    steps = 5 if args.sanity_check else -1
    train_classifier(max_steps=steps)
