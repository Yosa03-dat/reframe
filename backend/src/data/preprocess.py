import pandas as pd
import ast
import re
import os
import numpy as np

def clean_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions if any (e.g. @user)
    text = re.sub(r'\@\w+|\#','', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_indices(idx_str):
    if pd.isna(idx_str):
        return []
    idx_str = str(idx_str).strip()
    if idx_str == '[n/a]' or idx_str == 'n/a' or idx_str == '':
        return []
    try:
        # Convert literal string like '[1, 2]' into Python list
        return ast.literal_eval(idx_str)
    except:
        # Fallback regex extraction if ast fails
        matches = re.findall(r'\d+', idx_str)
        return [int(m) for m in matches]

def parse_responses(resp_str):
    if pd.isna(resp_str):
        return []
    resp_str = str(resp_str).strip()
    try:
        parsed = ast.literal_eval(resp_str)
        if isinstance(parsed, list):
            return [clean_text(r) for r in parsed]
        elif isinstance(parsed, str):
            return [clean_text(parsed)]
        return []
    except:
        return [clean_text(resp_str)]

def extract_lines(text):
    if pd.isna(text):
        return {}
    # Find all lines starting with number. e.g. "1. some text"
    lines = {}
    # Use regex to find lines starting with number dot.
    # The split might be tricky if numbers are in text, so we split by newlines or match the pattern
    # The format looks like: "1. 39869714\n" -> this is the ID?
    # Wait, looking at the exploration output:
    # id column: "1. 39869714\n2. \t39848775"
    # text column: "1. i joined gab to remind myself...\n2. another line"
    # It's better to find all matches of `r'^(\d+)\.\s*(.*)'` with MULTILINE flag.
    matches = re.finditer(r'^\s*(\d+)\.\s*(.*?)(?=(^\s*\d+\.\s*|\Z))', text, flags=re.MULTILINE | re.DOTALL)
    for match in matches:
        line_num = int(match.group(1))
        content = match.group(2).strip()
        lines[line_num] = content
    return lines

def process_datasets(raw_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    
    datasets = ['gab.csv', 'reddit.csv']
    
    classifier_data = []
    generator_data = []
    
    for ds_name in datasets:
        file_path = os.path.join(raw_dir, ds_name)
        print(f"Processing {file_path}...")
        
        try:
            # We use 'utf-8' with replacement for bad characters, or fallback to latin1
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
                
            # Filter to required columns, handles reddit having 32 cols
            cols_to_keep = ['id', 'text', 'hate_speech_idx', 'response']
            df = df[cols_to_keep]
            
            for _, row in df.iterrows():
                text_col = row['text']
                hate_idx = parse_indices(row['hate_speech_idx'])
                responses = parse_responses(row['response'])
                
                lines_dict = extract_lines(text_col)
                
                # If extraction fails but text exists, maybe it's a single line without "1. " prefix
                if not lines_dict and pd.notna(text_col) and len(str(text_col).strip()) > 0:
                    lines_dict = {1: str(text_col).strip()}
                
                for line_num, line_text in lines_dict.items():
                    cleaned_line = clean_text(line_text)
                    if not cleaned_line:
                        continue
                        
                    is_toxic = 1 if line_num in hate_idx else 0
                    
                    # Add to classifier dataset
                    classifier_data.append({
                        'text': cleaned_line,
                        'label': is_toxic,
                        'source': ds_name
                    })
                    
                    # Add to generator dataset if toxic and has responses
                    if is_toxic and len(responses) > 0:
                        generator_data.append({
                            'toxic_text': cleaned_line,
                            'responses': responses, # Keeping as list for flexibility
                            'source': ds_name
                        })
                        
        except Exception as e:
            print(f"Error processing {ds_name}: {e}")
            
    # Save datasets
    classifier_df = pd.DataFrame(classifier_data)
    generator_df = pd.DataFrame(generator_data)
    
    classifier_out = os.path.join(processed_dir, 'classifier_dataset.csv')
    generator_out = os.path.join(processed_dir, 'generator_dataset.csv')
    
    classifier_df.to_csv(classifier_out, index=False)
    generator_df.to_csv(generator_out, index=False)
    
    print(f"Saved {len(classifier_df)} samples to {classifier_out}")
    print(f"Saved {len(generator_df)} samples to {generator_out}")

if __name__ == "__main__":
    raw_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw')
    processed_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed')
    process_datasets(raw_directory, processed_directory)
