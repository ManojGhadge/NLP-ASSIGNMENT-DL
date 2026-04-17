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
