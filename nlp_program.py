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

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils import resample

nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv('TweetSentiment.csv', encoding='ISO-8859-1')

# Task 1: EDA
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('eda_sentiment_distribution.png', dpi=300)
plt.close()

df['text_length'] = df['text'].str.len()
plt.figure(figsize=(6,4))
sns.histplot(df['text_length'], bins=30)
plt.title('Tweet Length Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.savefig('eda_length_distribution.png', dpi=300)
plt.close()

all_text = ' '.join(df['text'].fillna('').astype(str).tolist())
wc = WordCloud(width=800, height=400, background_color='white', stopwords=set(stopwords.words('english')))
wc.generate(all_text)
wc.to_file('eda_wordcloud.png')

# Task 2: Preprocessing
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

df['clean_text'] = df['text'].fillna('').astype(str).apply(clean_text)

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_tokens(text):
    tokens = nltk.word_tokenize(text)
    filtered = [ps.stem(w) for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(filtered)

df['processed'] = df['clean_text'].apply(preprocess_tokens)


# Task 3: Modeling
X = df['processed']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_df = pd.concat([X_train, y_train], axis=1)
max_size = train_df['sentiment'].value_counts().max()
resampled = []
for sentiment, group in train_df.groupby('sentiment'):
    resampled.append(group.sample(max_size, replace=True, random_state=42))
train_resampled = pd.concat(resampled)
X_train = train_resampled['processed']
y_train = train_resampled['sentiment']


# 3.1 Naive Bayes
vectorizer_nb = TfidfVectorizer(max_features=5000)
X_train_nb = vectorizer_nb.fit_transform(X_train)
X_test_nb = vectorizer_nb.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_nb, y_train)

y_pred_nb = nb_model.predict(X_test_nb)
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))


# 3.2 FFNN
vectorizer_nn = TfidfVectorizer(max_features=5000)
X_train_nn = vectorizer_nn.fit_transform(X_train).toarray()
X_test_nn = vectorizer_nn.transform(X_test).toarray()

model_ffnn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_nn.shape[1],)),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
model_ffnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_ffnn.fit(X_train_nn, y_train.map({'negative':0,'neutral':1,'positive':2}).values,
               epochs=5, batch_size=32, validation_split=0.1)

y_pred_nn = model_ffnn.predict(X_test_nn)
y_pred_nn_labels = np.argmax(y_pred_nn, axis=1)
print("FFNN Classification Report:\n", classification_report(y_test.map({'negative':0,'neutral':1,'positive':2}).values, y_pred_nn_labels))
print("Confusion Matrix:\n", confusion_matrix(y_test.map({'negative':0,'neutral':1,'positive':2}).values, y_pred_nn_labels))

# 3.3 Binary Classification
bin_df = df[df['sentiment'] != 'neutral']
Xb = bin_df['processed']; yb = bin_df['sentiment']
Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=42)

vec_bin = TfidfVectorizer(max_features=3000)
Xb_train_b = vec_bin.fit_transform(Xb_train);
Xb_test_b = vec_bin.transform(Xb_test)

nb_bin = MultinomialNB()
nb_bin.fit(Xb_train_b, yb_train)

yb_pred = nb_bin.predict(Xb_test_b)
print("Binary NB Report:\n", classification_report(yb_test, yb_pred))

# Task 4: Semantic Text Similarity
embeddings = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        vals = line.split()
        word = vals[0]; vec = np.array(vals[1:], dtype='float32')
        embeddings[word] = vec

def sentence_vector(sent):
    words = sent.split()
    vecs = [embeddings[w] for w in words if w in embeddings]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

def cosine_sim(v1, v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

pos_texts = df[df['sentiment']=='positive']['processed'].sample(15, random_state=42).tolist()
vecs = [sentence_vector(s) for s in pos_texts]

similarities = []
for i in range(5):
    sim = cosine_sim(vecs[i], vecs[i+1])
    similarities.append((pos_texts[i], pos_texts[i+1], sim))

for a, b, sim in similarities:
    print(f"Sentence A: {a}\nSentence B: {b}\nSimilarity: {sim:.4f}\n")
