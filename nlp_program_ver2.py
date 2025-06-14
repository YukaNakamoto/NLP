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
from textblob import TextBlob  # for optional spell correction
from transformers import BertTokenizer, BertModel  # import BERT base model for embeddings
from sklearn.metrics.pairwise import cosine_similarity
import torch


# Download required NLTK data files
nltk.download('stopwords')
nltk.download('punkt')  # tokenizer data


# Load CSV of tweets with ISO-8859-1 encoding to handle special characters
df = pd.read_csv('TweetSentiment.csv', encoding='ISO-8859-1')

# Task 1: Exploratory Data Analysis 
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
    # stopwords=set()
)
wc.generate(all_text)
wc.to_file('eda_wordcloud.png')


# Task 2: Preprocessing
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

'''
def clean_text(text):
    """
    Clean tweet text by removing mentions, hashtags, URLs, lowercasing, and trimming whitespace.
    """
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)      # remove @mentions
    text = re.sub(r'#\w+', '', text)               # remove hashtags
    text = re.sub(r'http\S+|www\.\S+', '', text) # remove URLs
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = text.lower().strip()                     # lowercase and strip
    text = re.sub(r"\s+", " ", text)           # collapse multiple spaces
    return text

# Apply cleaning to original tweets
df['clean_text'] = df['text'].fillna('').astype(str).apply(clean_text)

# Prepare for tokenization: stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def correct_spelling(text):
    """
    TextBlob によるスペル補正を行う（オプション）。
    処理が重いので、必要な場合のみ適用する。
    """
    try:
        return str(TextBlob(text).correct())
    except Exception:
        return text  # 補正に失敗したら元のテキストを返す

# Apply cleaning
df['clean_text'] = df['text'].fillna('').astype(str).apply(clean_text)

# ここでスペル補正を挟みたい場合は True に
USE_SPELL_CORRECTION = True
if USE_SPELL_CORRECTION:
    df['clean_text'] = df['clean_text'].apply(correct_spelling)

# Tokenization + stopword removal + stemming
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_tokens(text):
    tokens = nltk.word_tokenize(text)    # split into words
    filtered = [ps.stem(w) for w in tokens if w.isalpha() and w not in stop_words]  # keep alphabetic tokens not in stopwords, then stem
    return ' '.join(filtered)

df['processed'] = df['clean_text'].apply(preprocess_tokens)
'''

# Task 3: Modeling
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


# 3.1 Naive Bayes Classification
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


# 3.2 Feed-Forward Neural Network (FFNN)
# Prepare TF-IDF features for NN as dense arrays
vectorizer_nn = TfidfVectorizer(max_features=5000)
X_train_nn = vectorizer_nn.fit_transform(X_train).toarray()
X_test_nn = vectorizer_nn.transform(X_test).toarray()
# map sentiment labels to numeric classes
label_map = {'negative':0,'neutral':1,'positive':2}

# define FFNN model
model_ffnn = Sequential([
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


# 3.3 Binary Classification (Positive vs. Negative)
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


# Task 4: Semantic Text Similarity using BERT embeddings
# 4.1) Initialize BERT tokenizer and model (using the base uncased model)
tokenizer_emb = BertTokenizer.from_pretrained('bert-base-uncased')
model_emb     = BertModel.from_pretrained('bert-base-uncased')
model_emb.eval()  # set to evaluation mode

# 4.2) Define a function to get the [CLS] embedding for a piece of text
def get_cls_embedding(text):
    """
    Tokenize input text, run through BERT, and return the [CLS] token embedding as a numpy array.
    """
    inputs = tokenizer_emb(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=128
    )
    with torch.no_grad():
        outputs = model_emb(**inputs)
    # outputs.last_hidden_state shape: (1, seq_len, hidden_size)
    # we take the first token ([CLS]) embedding
    cls_vec = outputs.last_hidden_state[0, 0]
    return cls_vec.cpu().numpy()

# 4.3) Sample 15 positive tweets from the processed column
pos_samples = (
    df[df['sentiment'] == 'positive']
      .dropna(subset=['processed'])['processed']
      .sample(15, random_state=42)
      .tolist()
)

# 4.4) Compute embeddings for each sampled sentence
embeddings = [get_cls_embedding(sent) for sent in pos_samples]

# 4.5) Compute and print cosine similarities for the first 5 consecutive pairs
print("Semantic Text Similarities (first 5 pairs):\n")
for i in range(5):
    sent_a, sent_b = pos_samples[i], pos_samples[i+1]
    sim_score = cosine_similarity(
        embeddings[i].reshape(1, -1),
        embeddings[i+1].reshape(1, -1)
    )[0,0]
    print(f"Sentence A: {sent_a}")
    print(f"Sentence B: {sent_b}")
    print(f"Cosine similarity: {sim_score:.4f}\n")

# Bonus 5) Define a simple compute_metrics function
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average='weighted')
    }

# Bonus 6) Initialize Trainer for multi-class
trainer_multi = Trainer(
    model=model_multi,
    args=train_args,
    train_dataset=multi_ds['train'],
    eval_dataset=multi_ds['test'],
    tokenizer=tokenizer_emb,
    compute_metrics=compute_metrics
)

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import numpy as np

# 1. Label encoding (sentiment to numeric)
label_list = ['negative', 'neutral', 'positive']
label2id = {label: idx for idx, label in enumerate(label_list)}
df['label'] = df['sentiment'].map(label2id)

# 2. Train/test split
train_df, test_df = train_test_split(df[['text', 'label']], test_size=0.2, stratify=df['label'], random_state=42)

# 3. Convert to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# 4. Tokenizer setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_batch(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_batch, batched=True)
test_dataset = test_dataset.map(tokenize_batch, batched=True)

# 5. Set format for PyTorch training
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 6. Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# 7. Define compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average='weighted')
    }

# 8. Define training arguments
training_args = TrainingArguments(
    output_dir='./bert_sentiment',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=50,
)

# 9. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 10. Train and evaluate
print("=== Fine-tuning BERT for Sentiment Classification ===")
trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)
