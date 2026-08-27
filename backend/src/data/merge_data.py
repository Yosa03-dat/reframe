import os
import pandas as pd
from preprocess import clean_text

def main():
    # Define paths based on current file location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_path = os.path.join(raw_dir, "train.csv")
    test_path = os.path.join(raw_dir, "test.csv")
    test_labels_path = os.path.join(raw_dir, "test_labels.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_labels_df = pd.read_csv(test_labels_path)
    
    print("Merging test and test_labels...")
    # Merge on 'id'
    merged_test_df = pd.merge(test_df, test_labels_df, on='id')
    
    print("Cleaning merged test data...")
    # Remove rows where 'toxic' == -1
    cleaned_test_df = merged_test_df[merged_test_df['toxic'] != -1]
    
    print("Concatenating train and cleaned test data...")
    # Concatenate train and cleaned test
    final_df = pd.concat([train_df, cleaned_test_df], ignore_index=True)
    
    print("Applying text preprocessing to 'comment_text'...")
    # Apply clean_text from preprocess.py
    final_df['text'] = final_df['comment_text'].apply(clean_text)
    
    # Create label column (using 'toxic' column as the label for ToxicityClassifierDataset)
    final_df['label'] = final_df['toxic']
    
    # Keep only necessary columns for the classifier
    classifier_df = final_df[['id', 'text', 'label']]
    
    # Filter out empty texts after cleaning
    classifier_df = classifier_df[classifier_df['text'].str.strip() != ""]
    
    # Save to processed directory
    output_path = os.path.join(processed_dir, "jigsaw_classifier_dataset.csv")
    print(f"Saving preprocessed dataset to {output_path}...")
    classifier_df.to_csv(output_path, index=False)
    
    print(f"Success! Total rows in preprocessed dataset: {len(classifier_df)}")

if __name__ == "__main__":
    main()
