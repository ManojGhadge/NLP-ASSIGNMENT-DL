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
#  First tries to download 20 Newsgroups from sklearn.
#  If it times out or fails, falls back to a small built-in
#  CSV dataset so you can still run the full pipeline.
# ============================================================

import signal
import threading

CATEGORIES = [
    'sci.space',
    'rec.sport.hockey',
    'talk.politics.misc',
    'comp.graphics'
]

# --- Spinner so you can see it's alive ---
class Spinner:
    def __init__(self, message="  Working"):
        self.message = message
        self.running = False
        self._thread = threading.Thread(target=self._spin)

    def _spin(self):
        chars = ['|', '/', '-', '\\']
        i = 0
        while self.running:
            print(f"\r{self.message} {chars[i % 4]}", end='', flush=True)
            i += 1
            import time; time.sleep(0.2)

    def start(self):
        self.running = True
        self._thread.start()

    def stop(self):
        self.running = False
        self._thread.join()
        print(f"\r{self.message} — done!          ")


# --- Try downloading with a 60-second timeout ---
train_data = None
test_data  = None
USE_FALLBACK = False

print("=" * 55)
print("  Loading dataset (timeout: 60s)...")
print("=" * 55)

spinner = Spinner("  Downloading 20 Newsgroups")
spinner.start()

try:
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(fetch_20newsgroups,
                             subset='train', categories=CATEGORIES,
                             remove=('headers','footers','quotes'),
                             random_state=42)
        f2 = executor.submit(fetch_20newsgroups,
                             subset='test', categories=CATEGORIES,
                             remove=('headers','footers','quotes'),
                             random_state=42)
        train_data = f1.result(timeout=60)
        test_data  = f2.result(timeout=60)
    spinner.stop()
    print("  20 Newsgroups loaded successfully.")

except Exception as e:
    spinner.stop()
    print(f"  Download failed or timed out: {e}")
    print("  Using built-in fallback dataset instead...\n")
    USE_FALLBACK = True


# ============================================================
#  FALLBACK DATASET
#  If the download fails, we use 200 hand-written samples
#  covering the same 4 categories. Small but enough to
#  demonstrate the full preprocessing → model → eval pipeline.
# ============================================================

if USE_FALLBACK:
    from sklearn.preprocessing import LabelEncoder

    fallback_texts = [
        # sci.space (label 0)
        "NASA launched a new rocket to the moon last week",
        "The space shuttle orbited Earth for six days",
        "Astronomers discovered a new exoplanet with a telescope",
        "Mars rover sends back photos of the red planet surface",
        "The Hubble telescope captured images of distant galaxies",
        "SpaceX successfully landed a reusable rocket booster",
        "Scientists detected gravitational waves from black holes",
        "Astronauts completed a spacewalk outside the station",
        "The satellite was placed in geostationary orbit",
        "Solar flares can disrupt GPS and radio communications",
        "NASA plans a crewed mission to Mars in the next decade",
        "The James Webb telescope reveals early universe images",
        "Orbital mechanics determines spacecraft trajectory paths",
        "Jupiter moon Europa may have liquid water beneath ice",
        "The Voyager probe is now in interstellar space",
        "Rocket propulsion systems use liquid hydrogen fuel",
        "Astronomers mapped the cosmic microwave background",
        "Space debris poses risk to operational satellites",
        "The ISS receives supplies from unmanned cargo ships",
        "Lunar samples contain evidence of ancient volcanism",
        "Telescope arrays detect radio signals from pulsars",
        "The asteroid belt lies between Mars and Jupiter orbits",
        "Comets are composed of ice and rocky material",
        "The sun releases energy through nuclear fusion",
        "Spacecraft use ion propulsion for deep space travel",
        "Satellites monitor weather patterns from orbit",
        "The Milky Way contains over 200 billion stars",
        "Astronomers study red giant star expansion phases",
        "Gravity assists allow probes to gain speed cheaply",
        "Cosmic rays are high energy particles from outer space",
        "Space tourism companies are developing passenger rockets",
        "The Falcon Heavy rocket carries large payloads to orbit",
        "Moon landing missions collected hundreds of rock samples",
        "Radio telescopes detect signals from distant quasars",
        "Solar wind particles create auroras near polar regions",
        "Black holes warp spacetime around their event horizons",
        "Orbital decay causes satellites to reenter atmosphere",
        "Nebulae are clouds of gas where new stars are born",
        "The Mars atmosphere is mostly carbon dioxide gas",
        "Pluto was reclassified as a dwarf planet in 2006",
        "Deep space probes require nuclear power sources",
        "Astronomers use spectroscopy to study star composition",
        "The Artemis program aims to return humans to the moon",
        "Cubesats are small low cost satellites for research",
        "Spacewalks require pressurized extravehicular suits",
        "The heliosphere protects the solar system from radiation",
        "Saturn rings are composed of ice and rock particles",
        "Lunar craters reveal the history of asteroid impacts",
        "Space stations require regular resupply missions",
        "Astronaut training includes zero gravity simulation",

        # rec.sport.hockey (label 1)
        "The hockey team won the championship after overtime",
        "He scored a hat trick in the third period tonight",
        "The goalie made thirty saves during last night game",
        "Hockey players wear pads helmets and skates for safety",
        "The NHL season begins in October each year",
        "The puck deflected off the post and went in the net",
        "Power play goals are scored when opponents are penalized",
        "Ice resurfacing machines clean the rink between periods",
        "The defenseman blocked the slap shot in front of goal",
        "Hockey sticks are made of carbon fiber or wood",
        "The Stanley Cup is the oldest professional sports trophy",
        "Face-off circles are located at center and both ends",
        "Overtime in playoffs continues until a goal is scored",
        "The winger skated down the boards and passed to center",
        "Penalty shots are awarded for certain rule violations",
        "Ice hockey originated in Canada during the 1800s",
        "The team practiced passing drills on the ice today",
        "Goaltenders use large pads to block low shots",
        "The captain wore the letter C on his jersey",
        "Hockey trades happen frequently before the deadline",
        "Players serve two minutes in the penalty box for tripping",
        "The arena sold out every home game this season",
        "Zamboni drivers maintain ice quality during breaks",
        "The rookie scored his first NHL goal on a breakaway",
        "Line changes happen quickly to keep players fresh",
        "Slap shots can reach speeds over 100 miles per hour",
        "The team pulled the goalie for an extra attacker",
        "Icing occurs when the puck is shot across both red lines",
        "Hockey commentators praised the defensive strategy",
        "The coach called a timeout to adjust the game plan",
        "The backup goalie started after the starter was injured",
        "A hat trick requires three goals from the same player",
        "Hockey fights result in five minute major penalties",
        "The power forward battled for pucks along the boards",
        "Draft picks are selected based on player potential",
        "Penalty kill units block shots to prevent power play goals",
        "The rink boards and glass surround the playing surface",
        "Teams travel by charter jet to away games",
        "The winning goal was reviewed by video officials",
        "Pre-game skates allow players to warm up on ice",
        "Hockey analytics track shot attempts corsi and fenwick",
        "The team celebrated their division title on home ice",
        "Stick handling drills improve puck control skills",
        "The franchise goalie signed a long term contract",
        "Hockey helmets must meet certified safety standards",
        "Offside occurs when an attacker enters zone before puck",
        "Players tape their sticks before every game",
        "The trade deadline brought several veteran players",
        "Playoff hockey is more physical than regular season",
        "The broadcast showed a slow motion replay of the goal",

        # talk.politics.misc (label 2)
        "The government passed a new healthcare reform bill",
        "Politicians debated the proposed tax increase policy",
        "The election results showed a divided electorate",
        "Congress voted on the annual federal budget yesterday",
        "The senator gave a speech about immigration reform",
        "Foreign policy decisions affect international relations",
        "Voters turned out in record numbers for the election",
        "The president signed an executive order on trade",
        "Political parties disagree on climate change policy",
        "The Supreme Court ruled on a constitutional rights case",
        "Lobbyists influence legislation through campaign donations",
        "The mayor proposed a new public transport bill",
        "Government spending increased for defense programs",
        "Civil rights legislation was a landmark achievement",
        "The opposition party rejected the proposed amendment",
        "Public opinion polls show declining approval ratings",
        "Trade tariffs affect domestic manufacturing industries",
        "The diplomat negotiated a bilateral treaty agreement",
        "Local elections determine city council membership",
        "Campaign finance reform is a contested political issue",
        "The referendum result surprised political analysts",
        "Welfare programs provide assistance to low income families",
        "The attorney general investigated corporate fraud cases",
        "Minimum wage debates divide political parties sharply",
        "Foreign aid budgets are often subject to political debate",
        "The governor declared a state of emergency after floods",
        "Political corruption scandals erode voter trust",
        "Term limits for legislators are debated frequently",
        "Healthcare costs are a key issue in every election",
        "The constitution protects free speech and assembly",
        "Tax reform affects both corporations and individuals",
        "The prime minister addressed parliament on the budget",
        "Bipartisan support is needed to pass major legislation",
        "The census determines congressional seat distribution",
        "Environmental regulations face opposition from industry",
        "Voter registration drives increase election participation",
        "The justice system faces calls for structural reform",
        "Defense spending accounts for a large budget portion",
        "International sanctions were imposed on the government",
        "The council debated zoning laws for the city center",
        "Social security funding is a long term fiscal challenge",
        "Political polarization makes compromise more difficult",
        "The ambassador was recalled following diplomatic tensions",
        "Municipal bonds fund local infrastructure development",
        "The whistleblower revealed classified government programs",
        "Immigration policy is debated across party lines",
        "The federal reserve raised interest rates this quarter",
        "Gerrymandering affects how congressional districts are drawn",
        "The committee held hearings on proposed legislation",
        "Presidential debates attract millions of television viewers",

        # comp.graphics (label 3)
        "OpenGL renders 3D graphics using a pipeline model",
        "The shader program runs on the GPU for fast rendering",
        "Texture mapping applies images onto 3D mesh surfaces",
        "Ray tracing simulates realistic light reflections",
        "The graphics card processes millions of polygons",
        "Antialiasing smooths jagged edges in rendered images",
        "Blender is a free open source 3D modeling software",
        "The vertex buffer stores geometry data for the GPU",
        "Pixel shaders calculate final color for each fragment",
        "Rasterization converts vector shapes to pixel grids",
        "Normal maps simulate fine surface detail without geometry",
        "The framebuffer stores the final rendered image output",
        "Ambient occlusion darkens crevices in 3D scenes",
        "Mesh topology affects deformation during animation",
        "Phong shading produces smooth lighting on curved surfaces",
        "The depth buffer prevents incorrect object overlap",
        "UV unwrapping maps 2D textures onto 3D surfaces",
        "Mipmaps improve performance for distant textured objects",
        "Deferred rendering separates geometry and lighting passes",
        "Procedural textures are generated mathematically",
        "The scene graph organizes objects in a hierarchical tree",
        "Skeletal animation uses bones to deform character meshes",
        "Post processing effects include bloom and motion blur",
        "Vulkan provides low level GPU access for developers",
        "Physics engines simulate collision and rigid body dynamics",
        "The projection matrix transforms 3D to 2D coordinates",
        "Bezier curves are used to create smooth vector paths",
        "Global illumination models indirect light bouncing",
        "Compute shaders perform parallel calculations on the GPU",
        "Level of detail reduces polygon count for distant objects",
        "Subsurface scattering simulates light through skin",
        "The renderer outputs images at 60 frames per second",
        "Particle systems simulate fire smoke and explosions",
        "Signed distance fields enable smooth font rendering",
        "Geometry shaders generate new primitives on the GPU",
        "Color grading adjusts the mood of rendered scenes",
        "Forward rendering processes each light per object",
        "Physically based rendering uses real world light equations",
        "The camera frustum defines the visible scene volume",
        "Tessellation subdivides geometry for smoother surfaces",
        "Occlusion culling skips rendering of hidden objects",
        "HDR imaging captures a wider range of brightness values",
        "Instancing renders many copies of a mesh efficiently",
        "The render pipeline begins with vertex transformation",
        "Shadow maps store depth from the light source view",
        "Image based lighting uses environment maps for reflections",
        "Anti aliasing techniques include MSAA FXAA and TAA",
        "Volumetric rendering simulates fog clouds and atmosphere",
        "Screen space reflections approximate mirror like surfaces",
        "The graphics pipeline ends with output merger stage",
    ]

    fallback_labels_str = (
        ['sci.space'] * 50 +
        ['rec.sport.hockey'] * 50 +
        ['talk.politics.misc'] * 50 +
        ['comp.graphics'] * 50
    )

    le = LabelEncoder()
    fallback_labels = le.fit_transform(fallback_labels_str)

    # Split into train / test
    from sklearn.model_selection import train_test_split as tts
    texts_train, texts_test, y_train, y_test = tts(
        fallback_texts, fallback_labels,
        test_size=0.25, random_state=42, stratify=fallback_labels
    )

    # Wrap in a simple namespace so the rest of the code works unchanged
    class DataBunch:
        def __init__(self, data, target, target_names):
            self.data         = data
            self.target       = target
            self.target_names = target_names

    train_data = DataBunch(texts_train, y_train, le.classes_.tolist())
    test_data  = DataBunch(texts_test,  y_test,  le.classes_.tolist())

    print(f"  Fallback dataset ready.")

else:
    y_train = train_data.target
    y_test  = test_data.target


print(f"\n  Training documents : {len(train_data.data)}")
print(f"  Testing  documents : {len(test_data.data)}")
print(f"  Categories         : {train_data.target_names}\n")

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

# y_train = train_data.target
# y_test  = test_data.target

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



# ============================================================
#  SECTION 10: FINAL SUMMARY
# ============================================================

print("=" * 55)
print("  FINAL RESULTS SUMMARY")
print("=" * 55)
for name, acc in results_summary.items():
    print(f"  {name:<35} → {acc*100:.2f}%")

print("\n  Output files saved in /outputs folder:")
print("    - confusion_matrix.png")
print("    - model_comparison.png")
print("\n  Assignment complete!")
