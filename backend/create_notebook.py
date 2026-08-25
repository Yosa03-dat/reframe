import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis (EDA)\n",
    "This notebook performs EDA on the processed datasets to answer critical questions about class imbalance, token length distribution, intervention response diversity, and lexical N-grams."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import ast\n",
    "import re\n",
    "from collections import Counter\n",
    "import plotly.express as px\n",
    "\n",
    "# Load datasets\n",
    "classifier_df = pd.read_csv('../data/processed/classifier_dataset.csv')\n",
    "generator_df = pd.read_csv('../data/processed/generator_dataset.csv')\n",
    "print(f\"Loaded {len(classifier_df)} classifier samples and {len(generator_df)} generator samples.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Class Imbalance Check"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class_counts = classifier_df['label'].value_counts().reset_index()\n",
    "class_counts.columns = ['label', 'count']\n",
    "class_counts['label_name'] = class_counts['label'].map({0: 'Non-Toxic (0)', 1: 'Toxic (1)'})\n",
    "\n",
    "fig = px.pie(class_counts, values='count', names='label_name', title='Class Imbalance (Toxic vs Non-Toxic)')\n",
    "fig.show()\n",
    "\n",
    "print(class_counts)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Token Length Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier_df['word_count'] = classifier_df['text'].fillna('').apply(lambda x: len(str(x).split()))\n",
    "\n",
    "fig2 = px.histogram(classifier_df, x='word_count', color='label', nbins=100, \n",
    "                    title='Input Text Word Count Distribution',\n",
    "                    labels={'word_count': 'Word Count', 'label': 'Toxicity Label'})\n",
    "fig2.update_layout(barmode='overlay')\n",
    "fig2.update_traces(opacity=0.75)\n",
    "fig2.show()\n",
    "\n",
    "print(\"Input Text Stats:\")\n",
    "print(classifier_df['word_count'].describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "generator_df['response_length'] = generator_df['response'].fillna('').apply(lambda x: len(str(x).split()))\n",
    "\n",
    "fig3 = px.histogram(generator_df, x='response_length', nbins=50, \n",
    "                    title='Human Response Word Count Distribution')\n",
    "fig3.show()\n",
    "\n",
    "print(\"Responses Stats:\")\n",
    "print(generator_df['response_length'].describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Intervention Response Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_responses = generator_df['response'].dropna().tolist()\n",
    "\n",
    "response_counter = Counter(all_responses)\n",
    "print(f\"Total human responses: {len(all_responses)}\")\n",
    "print(f\"Unique human responses: {len(response_counter)}\")\n",
    "print(\"\\nTop 10 most common responses:\")\n",
    "top_10 = response_counter.most_common(10)\n",
    "for r, c in top_10:\n",
    "    print(f\"{c} times: {r[:100]}...\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Lexical and N-Gram Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import string\n",
    "from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS\n",
    "\n",
    "toxic_texts = classifier_df[classifier_df['label'] == 1]['text'].dropna().tolist()\n",
    "\n",
    "def get_ngrams(text_list, n):\n",
    "    ngrams = []\n",
    "    for text in text_list:\n",
    "        words = str(text).translate(str.maketrans('', '', string.punctuation)).split()\n",
    "        words = [w for w in words if w.strip() and w.lower() not in ENGLISH_STOP_WORDS]\n",
    "        for i in range(len(words)-n+1):\n",
    "            ngrams.append(\" \".join(words[i:i+n]))\n",
    "    return Counter(ngrams)\n",
    "\n",
    "bigrams = get_ngrams(toxic_texts, 2)\n",
    "trigrams = get_ngrams(toxic_texts, 3)\n",
    "\n",
    "top_bigrams_df = pd.DataFrame(bigrams.most_common(15), columns=['Bigram', 'Count'])\n",
    "fig4 = px.bar(top_bigrams_df, x='Count', y='Bigram', orientation='h', title='Top 15 Toxic Bigrams')\n",
    "fig4.update_layout(yaxis={'categoryorder':'total ascending'})\n",
    "fig4.show()\n",
    "\n",
    "top_trigrams_df = pd.DataFrame(trigrams.most_common(15), columns=['Trigram', 'Count'])\n",
    "fig5 = px.bar(top_trigrams_df, x='Count', y='Trigram', orientation='h', title='Top 15 Toxic Trigrams')\n",
    "fig5.update_layout(yaxis={'categoryorder':'total ascending'})\n",
    "fig5.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

os.makedirs('c:/Yoosuf/Sabra/Semester IV/Artificial Intelligence/Final_Project/Reframe/backend/notebooks', exist_ok=True)
with open('c:/Yoosuf/Sabra/Semester IV/Artificial Intelligence/Final_Project/Reframe/backend/notebooks/01_eda_and_cleaning.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
print("Notebook created successfully!")
