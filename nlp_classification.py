# ============================================================
#  SECTION 1: IMPORTS
#  All libraries needed for the entire project
# ============================================================

import re
import os
import warnings
import nltk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # prevents GUI popup errors on some systems
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

warnings.filterwarnings('ignore')



# ============================================================
#  SECTION 2: DOWNLOAD NLTK DATA
#  Downloads language resources NLTK needs (only on first run)
#  - punkt        : tokenizer model
#  - stopwords    : list of common words to remove (the, is, at…)
#  - wordnet      : vocabulary database used by lemmatizer
#  - punkt_tab    : updated tokenizer tables (required in newer NLTK)
# ============================================================

print("=" * 55)
print("  Downloading NLTK resources...")
print("=" * 55)

nltk.download('punkt',                    quiet=True)
nltk.download('stopwords',               quiet=True)
nltk.download('wordnet',                 quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab',               quiet=True)

print("  NLTK resources ready.\n")



# ============================================================
#  SECTION 3: LOAD THE DATASET
#  We use the built-in 20 Newsgroups dataset from scikit-learn.
#  It contains ~18,000 newsgroup posts across 20 topics.
#  We pick 4 categories to keep training fast on a CPU.
#
#  remove=('headers','footers','quotes') strips metadata so the
#  model learns from actual content, not sender names / dates.
# ============================================================

print("=" * 55)
print("  Loading 20 Newsgroups dataset...")
print("=" * 55)

CATEGORIES = [
    'sci.space',
    'rec.sport.hockey',
    'talk.politics.misc',
    'comp.graphics'
]

train_data = fetch_20newsgroups(
    subset='train',
    categories=CATEGORIES,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

test_data = fetch_20newsgroups(
    subset='test',
    categories=CATEGORIES,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

print(f"  Training documents : {len(train_data.data)}")
print(f"  Testing  documents : {len(test_data.data)}")
print(f"  Categories         : {train_data.target_names}\n")

# Show a raw sample so you can see what we start with
print("--- RAW sample (before preprocessing) ---")
print(train_data.data[0][:400])
print()






# ============================================================
#  SECTION 4: NLP PREPROCESSING FUNCTIONS
#
#  preprocess_text() runs each document through 5 stages:
#
#  Stage 1 – Lowercase
#    "Running" and "running" become the same token.
#
#  Stage 2 – Remove non-alphabetic characters
#    Strips numbers, punctuation, URLs, special symbols.
#
#  Stage 3 – Tokenization
#    Splits "I love NLP" → ["I", "love", "NLP"]
#    Uses NLTK's punkt tokenizer, which handles edge cases
#    like "Mr." or "U.S.A." better than a simple split().
#
#  Stage 4 – Stopword removal
#    Removes words like "the", "is", "at", "which", "on".
#    These appear in every document and carry no class signal.
#    Also drops tokens shorter than 3 characters.
#
#  Stage 5 – Stemming OR Lemmatization (one at a time)
#    Stemming    : chops word endings by rule → "running" → "run"
#                  Fast but sometimes produces non-real words
#                  e.g. "studies" → "studi"
#    Lemmatization: looks up the actual root in a dictionary
#                  → "better" → "good", "ran" → "run"
#                  Slower but produces real words.
#
#  We demonstrate both below for comparison.
# ============================================================

STOP_WORDS = set(stopwords.words('english'))
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text, use_stemming=False, use_lemmatization=True):
    """
    Clean and normalize a raw text document.

    Parameters
    ----------
    text              : str  — raw input document
    use_stemming      : bool — apply Porter Stemmer if True
    use_lemmatization : bool — apply WordNet Lemmatizer if True (default)

    Returns
    -------
    str — space-joined cleaned tokens
    """
    # Stage 1: lowercase everything
    text = text.lower()

    # Stage 2: keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Stage 3: tokenize
    tokens = word_tokenize(text)

    # Stage 4: remove stopwords and very short tokens
    tokens = [
        token for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    # Stage 5: reduce to root form
    if use_stemming:
        tokens = [stemmer.stem(token) for token in tokens]
    elif use_lemmatization:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)



# ============================================================
#  SECTION 5: APPLY PREPROCESSING TO ALL DOCUMENTS
#  This loops over every training and test document.
#  On a CPU this takes ~30-90 seconds depending on your machine.
# ============================================================

print("=" * 55)
print("  Preprocessing documents (this takes ~1 minute)...")
print("=" * 55)

X_train_clean = [preprocess_text(doc) for doc in train_data.data]
X_test_clean  = [preprocess_text(doc) for doc in test_data.data]

y_train = train_data.target
y_test  = test_data.target

# Show the same sample after preprocessing so the difference is visible
print("\n--- CLEANED sample (after preprocessing) ---")
print(X_train_clean[0][:400])
print()

# Quick demonstration of stemming vs lemmatization on a sample sentence
demo_sentence = "The researchers are studying machine learning algorithms"
print("--- Preprocessing Demo ---")
print(f"Original     : {demo_sentence}")
print(f"Lemmatized   : {preprocess_text(demo_sentence, use_lemmatization=True)}")
print(f"Stemmed      : {preprocess_text(demo_sentence, use_stemming=True)}")
print()
