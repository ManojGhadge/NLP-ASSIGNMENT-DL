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

