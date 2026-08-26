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

## Model Training

To train the Toxicity Classifier (DistilRoBERTa) and Intervention Generator (FLAN-T5) models, run the following scripts:

```bash
# Train the Classifier
python src/training/train_classifier.py

# Train the Generator
python src/training/train_generator.py
```
*Note: The best models are saved in the `backend/results/classifier/best_model` and `backend/results/generator/best_model` directories respectively. These directories are ignored by Git.*

## Running the API

Once the models are trained, you can run the highly optimized FastAPI inference server. The API pipeline dynamically loads the models using PyTorch FP16 precision on the GPU for sub-100ms response times.

```bash
# From the backend directory
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Once running, navigate to [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to explore the interactive API Swagger UI.

**Example Request:**
```bash
curl -X 'POST' \
  'http://localhost:8000/analyze' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Your argument is completely idiotic."
}'
```
