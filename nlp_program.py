# -*- coding: utf-8 -*-
"""
NLP Project 1.5: Twitter Sentiment Analysis Pipeline
- Task1: Exploratory Data Analysis (EDA)
- Task2: Preprocessing
- Task3: Modeling (Naive Bayes, FFNN, Binary Classification)
- Task4: Semantic Text Similarity

Requirements:
- Python 3.x
- pandas, numpy, matplotlib, seaborn, nltk, sklearn, wordcloud, tensorflow, transformers
"""

import pandas as pd                          # for data manipulation
import numpy as np                           # for numerical operations
import re                                    # for regular expressions
import matplotlib.pyplot as plt             # for plotting
import seaborn as sns                       # for statistical visualizations
from wordcloud import WordCloud             # for generating word clouds
from sklearn.model_selection import train_test_split  # to split data into train/test sets
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # to convert text to numeric features
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # evaluation metrics
import nltk                                  # natural language toolkit
from nltk.corpus import stopwords           # stopword list
from nltk.stem import PorterStemmer         # stemming
import tensorflow as tf                     # for neural network modeling
from tensorflow.keras import Sequential     # sequential model API
from tensorflow.keras.layers import Dense, Dropout  # layers for FFNN
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments  # for BERT model
from sklearn.utils import resample           # for handling class imbalance

# Download required NLTK data files
nltk.download('stopwords')
nltk.download('punkt')  # tokenizer data

#-------------------------------------------
# Load dataset
#-------------------------------------------
# Load CSV of tweets with ISO-8859-1 encoding to handle special characters
df = pd.read_csv('TweetSentiment.csv', encoding='ISO-8859-1')

#-------------------------------------------
# Task 1: Exploratory Data Analysis (EDA)
#-------------------------------------------
# Plot distribution of sentiment labels
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('eda_sentiment_distribution.png', dpi=300)
plt.close()

# Compute length of each tweet and plot histogram
df['text_length'] = df['text'].str.len()
plt.figure(figsize=(6,4))
sns.histplot(df['text_length'], bins=30)
plt.title('Tweet Length Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.savefig('eda_length_distribution.png', dpi=300)
plt.close()

# Generate a word cloud of all tweets (excluding English stopwords)
all_text = ' '.join(df['text'].fillna('').astype(str).tolist())
wc = WordCloud(
    width=800, height=400,
    background_color='white',
    stopwords=set(stopwords.words('english'))
)
wc.generate(all_text)
wc.to_file('eda_wordcloud.png')

#-------------------------------------------
# Task 2: Preprocessing
#-------------------------------------------

def clean_text(text):
    """
    Clean tweet text by removing mentions, hashtags, URLs, lowercasing, and trimming whitespace.
    """
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)      # remove @mentions
    text = re.sub(r'#\w+', '', text)               # remove hashtags
    text = re.sub(r'http\S+|www\.\S+', '', text) # remove URLs
    text = text.lower().strip()                     # lowercase and strip
    text = re.sub(r"\s+", " ", text)           # collapse multiple spaces
    return text

# Apply cleaning to original tweets
df['clean_text'] = df['text'].fillna('').astype(str).apply(clean_text)

# Prepare for tokenization: stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_tokens(text):
    """
    Tokenize, filter non-alphabetic words and stopwords, then apply stemming.
    Returns re-joined string of processed tokens.
    """
    tokens = nltk.word_tokenize(text)               # split into words
    # keep alphabetic tokens not in stopwords, then stem
    filtered = [ps.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(filtered)

# Create final processed text column
df['processed'] = df['clean_text'].apply(preprocess_tokens)

#-------------------------------------------
# Task 3: Modeling
#-------------------------------------------

# Split features (X) and labels (y)
X = df['processed']
y = df['sentiment']
# train/test split with stratification to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Address class imbalance by oversampling minority classes
train_df = pd.concat([X_train, y_train], axis=1)
max_size = train_df['sentiment'].value_counts().max()
resampled = []
for sentiment, group in train_df.groupby('sentiment'):
    # sample with replacement to match the largest class size
    resampled.append(group.sample(max_size, replace=True, random_state=42))
train_resampled = pd.concat(resampled)
# update X_train and y_train after balancing
X_train = train_resampled['processed']
y_train = train_resampled['sentiment']

#-------------------------------------------
# 3.1 Naive Bayes Classification
#-------------------------------------------
# Convert text to TF-IDF features with a limit on vocabulary size
vectorizer_nb = TfidfVectorizer(max_features=5000)
X_train_nb = vectorizer_nb.fit_transform(X_train)
X_test_nb = vectorizer_nb.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_nb, y_train)               # train model
# predict on test set
y_pred_nb = nb_model.predict(X_test_nb)
# output performance metrics
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))

#-------------------------------------------
# 3.2 Feed-Forward Neural Network (FFNN)
#-------------------------------------------
# Prepare TF-IDF features for NN as dense arrays
vectorizer_nn = TfidfVectorizer(max_features=5000)
X_train_nn = vectorizer_nn.fit_transform(X_train).toarray()
X_test_nn = vectorizer_nn.transform(X_test).toarray()
# map sentiment labels to numeric classes
label_map = {'negative':0,'neutral':1,'positive':2}
\ nmodel_ffnn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_nn.shape[1],)),  # hidden layer
    Dropout(0.5),                                                      # dropout for regularization
    Dense(3, activation='softmax')                                     # output layer for 3 classes
])
model_ffnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# train the FFNN with a validation split to monitor overfitting
model_ffnn.fit(
    X_train_nn,
    y_train.map(label_map).values,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)
# inference and evaluation
y_pred_nn = model_ffnn.predict(X_test_nn)
y_pred_nn_labels = np.argmax(y_pred_nn, axis=1)
print("FFNN Classification Report:\n", classification_report(y_test.map(label_map).values, y_pred_nn_labels))
print("Confusion Matrix:\n", confusion_matrix(y_test.map(label_map).values, y_pred_nn_labels))

#-------------------------------------------
# 3.3 Binary Classification (Positive vs. Negative)
#-------------------------------------------
# filter out neutral tweets
bin_df = df[df['sentiment'] != 'neutral']
Xb = bin_df['processed']
yb = bin_df['sentiment']
# split for binary task
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.2, stratify=yb, random_state=42
)
vec_bin = TfidfVectorizer(max_features=3000)
Xb_train_b = vec_bin.fit_transform(Xb_train)
Xb_test_b = vec_bin.transform(Xb_test)
nb_bin = MultinomialNB()
nb_bin.fit(Xb_train_b, yb_train)
yb_pred = nb_bin.predict(Xb_test_b)
print("Binary NB Report:\n", classification_report(yb_test, yb_pred))

#-------------------------------------------
# Task 4: Semantic Text Similarity
#-------------------------------------------
# Load GloVe embeddings from file into a dictionary
embeddings = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        vals = line.split()
        word = vals[0]
        vec = np.array(vals[1:], dtype='float32')
        embeddings[word] = vec

def sentence_vector(sent):
    """
    Compute average GloVe vector for a sentence.
    Returns zero vector if no known words found.
    """
    words = sent.split()
    vecs = [embeddings[w] for w in words if w in embeddings]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

def cosine_sim(v1, v2):
    """
    Compute cosine similarity between two vectors.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Sample a few positive tweets and compute pairwise similarity
pos_texts = df[df['sentiment']=='positive']['processed'].sample(15, random_state=42).tolist()
vecs = [sentence_vector(s) for s in pos_texts]
similarities = []
for i in range(5):
    sim = cosine_sim(vecs[i], vecs[i+1])
    similarities.append((pos_texts[i], pos_texts[i+1], sim))
# display results
for a, b, sim in similarities:
    print(f"Sentence A: {a}\nSentence B: {b}\nSimilarity: {sim:.4f}\n")
