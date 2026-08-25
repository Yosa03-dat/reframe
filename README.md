# Reframe Project

Reframe is an API designed to be integrated into community platforms (like Discord or a custom web app). It doesn't just quietly block toxic language; it returns an explanation of why the language was flagged to educate the user before they hit send.

The backend service and deep learning models are located in the `backend/` directory.

## Data Source
Dataset: [A Benchmark Dataset for Learning to Intervene in Online Hate Speech](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/tree/master/data)

## Setup Instructions

1. **Navigate to the Backend Directory:**
   ```bash
   cd backend
   ```

2. **Create Virtual Environment:**
   ```bash
   py -3.12 -m venv venv
   ```

3. **Activate the Environment:**
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Preprocessing the Data

Ensure the raw datasets (`gab.csv` and `reddit.csv`) are placed in `backend/data/raw/`. 

Then, run the preprocessing script to clean the data, parse the conversation threads, and generate the classifier and generator datasets:
```bash
cd backend
python src/data/preprocess.py
```
This will output `classifier_dataset.csv` and `generator_dataset.csv` into `backend/data/processed/`.
