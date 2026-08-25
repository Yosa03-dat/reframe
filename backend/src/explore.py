import pandas as pd
import ast
import re

def explore_csv(filepath):
    print(f"Exploring {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1')
    
    print(df.head(3))
    print(df['hate_speech_idx'].head(10).tolist())
    print("-" * 50)

explore_csv('data/raw/gab.csv')
explore_csv('data/raw/reddit.csv')
