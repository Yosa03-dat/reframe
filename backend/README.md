# Reframe Backend API

Reframe is an API designed to be integrated into community platforms (like Discord or a custom web app). It doesn't just quietly block toxic language; it returns an explanation of why the language was flagged to educate the user before they hit send.

## Data Source
Dataset: [A Benchmark Dataset for Learning to Intervene in Online Hate Speech](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/tree/master/data)

## Setup Instructions

1. **Create Virtual Environment:**
   ```bash
   py -3.12 -m venv venv
   ```

2. **Activate the Environment:**
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Preprocessing the Data

Ensure the raw datasets (`gab.csv` and `reddit.csv`) are placed in `data/raw/`. 

Then, run the preprocessing script to clean the data, parse the conversation threads, and generate the classifier and generator datasets:
```bash
python src/data/preprocess.py
```
This will output `classifier_dataset.csv` and `generator_dataset.csv` into `data/processed/`.
