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


# ============================================================
#  SECTION 6: TEXT VECTORIZATION
#
#  Machine learning models cannot read text — they need numbers.
#  Vectorization converts each document into a numeric vector.
#
#  CountVectorizer
#    Builds a vocabulary of the top N words across all documents.
#    Each document becomes a vector of word counts.
#    Document "I love space space" → [0, 0, 1, 2, ...] (love=1, space=2)
#    Problem: frequent words like "said" dominate even if not meaningful.
#
#  TF-IDF (Term Frequency – Inverse Document Frequency)
#    TF  = how often a word appears in THIS document
#    IDF = penalizes words that appear in MANY documents (less unique)
#    Result: rare, specific words get higher scores.
#    "NASA" in a space article → high score
#    "said" in every article → low score
#
#  max_features=5000  : keep only top 5000 words (speeds up training)
#  ngram_range=(1,2)  : include single words AND two-word phrases
#                       e.g. "machine learning" as one feature
#  sublinear_tf=True  : applies log(1 + tf) to dampen very high counts
# ============================================================

print("=" * 55)
print("  Vectorizing text...")
print("=" * 55)

# --- CountVectorizer ---
count_vec = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_count = count_vec.fit_transform(X_train_clean)   # learn vocab + transform
X_test_count  = count_vec.transform(X_test_clean)         # transform only (no refit)

# --- TF-IDF Vectorizer ---
tfidf_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = tfidf_vec.fit_transform(X_train_clean)
X_test_tfidf  = tfidf_vec.transform(X_test_clean)

print(f"  CountVectorizer matrix : {X_train_count.shape}  (documents × features)")
print(f"  TF-IDF matrix          : {X_train_tfidf.shape}  (documents × features)")

# Show 10 sample features
features = tfidf_vec.get_feature_names_out()
print(f"  Sample features: {list(features[:5])} ... {list(features[-5:])}\n")




# ============================================================
#  SECTION 7: TRAIN AND EVALUATE MODELS
#
#  We test 3 combinations:
#    1. Naive Bayes   + CountVectorizer
#    2. Naive Bayes   + TF-IDF
#    3. Logistic Regression + TF-IDF
#
#  Naive Bayes
#    Fast probabilistic classifier. Works very well for text.
#    Assumes each word is independent (the "naive" assumption).
#    alpha=0.1 is Laplace smoothing — prevents zero probabilities
#    for words not seen during training.
#
#  Logistic Regression
#    Learns a weight for each feature (word).
#    Often the strongest baseline for text classification.
#    max_iter=1000 ensures it converges properly.
#
#  Metrics reported:
#    Precision : of all predicted "sci.space", how many were really space?
#    Recall    : of all actual "sci.space" docs, how many did we catch?
#    F1-score  : harmonic mean of precision and recall
#    Accuracy  : overall percentage correct
# ============================================================

results_summary = {}   # store accuracies for the comparison chart


def train_and_evaluate(model, X_tr, X_te, y_tr, y_te, model_name, vec_name):
    """Train model, print full report, return predictions and accuracy."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)

    label = f"{model_name} + {vec_name}"
    results_summary[label] = acc

    print(f"\n{'=' * 55}")
    print(f"  {label}")
    print(f"{'=' * 55}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)\n")
    print(classification_report(
        y_te, y_pred,
        target_names=train_data.target_names,
        digits=4
    ))
    return y_pred


print("=" * 55)
print("  Training and evaluating models...")
print("=" * 55)

# Experiment 1
preds_nb_count = train_and_evaluate(
    MultinomialNB(alpha=0.1),
    X_train_count, X_test_count, y_train, y_test,
    "Naive Bayes", "CountVectorizer"
)

# Experiment 2
preds_nb_tfidf = train_and_evaluate(
    MultinomialNB(alpha=0.1),
    X_train_tfidf, X_test_tfidf, y_train, y_test,
    "Naive Bayes", "TF-IDF"
)

# Experiment 3
preds_lr_tfidf = train_and_evaluate(
    LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    X_train_tfidf, X_test_tfidf, y_train, y_test,
    "Logistic Regression", "TF-IDF"
)




# ============================================================
#  SECTION 8: VISUALIZATIONS
#  Saves two PNG files to your project folder:
#    confusion_matrix.png  — shows where the best model makes errors
#    model_comparison.png  — bar chart comparing all 3 experiments
# ============================================================

os.makedirs('outputs', exist_ok=True)

# --- Plot 1: Confusion Matrix for best model (LR + TF-IDF) ---
cm = confusion_matrix(y_test, preds_lr_tfidf)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=train_data.target_names,
    yticklabels=train_data.target_names
)
plt.title('Confusion Matrix — Logistic Regression + TF-IDF', fontsize=13)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()
print("\n  Saved: outputs/confusion_matrix.png")

# --- Plot 2: Model comparison bar chart ---
plt.figure(figsize=(8, 5))
colors = ['#5563D4', '#3BA87B', '#E8593C']
bars   = plt.bar(results_summary.keys(), results_summary.values(), color=colors, width=0.5)
plt.ylim(0.70, 1.0)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracy Comparison', fontsize=13)
plt.xticks(fontsize=10)

for bar, val in zip(bars, results_summary.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.004,
        f'{val:.3f}',
        ha='center', va='bottom', fontsize=11, fontweight='bold'
    )

plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150)
plt.close()
print("  Saved: outputs/model_comparison.png")



# ============================================================
#  SECTION 9: PREDICT ON NEW CUSTOM TEXT
#  Test the trained model on your own sentences
# ============================================================

print("\n" + "=" * 55)
print("  Testing on custom sentences")
print("=" * 55)

custom_texts = [
    "NASA launched a new rocket to explore Mars and Jupiter",
    "The hockey team scored three goals in the final period",
    "The government passed a new bill about tax reform",
    "OpenGL rendering pipeline and 3D graphics shaders"
]

# Use the best model: LR + TF-IDF
lr_best = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
lr_best.fit(X_train_tfidf, y_train)

for sentence in custom_texts:
    cleaned   = preprocess_text(sentence)
    vectorized = tfidf_vec.transform([cleaned])
    prediction = lr_best.predict(vectorized)[0]
    confidence = lr_best.predict_proba(vectorized).max()
    category   = train_data.target_names[prediction]
    print(f"  Input      : {sentence}")
    print(f"  Predicted  : {category}  (confidence: {confidence:.2%})\n")
