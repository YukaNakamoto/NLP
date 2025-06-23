import pandas as pd
import numpy as np
import re
import inflect
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler, FunctionTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity

# Setup & Data Loading
# Download necessary NLTK resources for tokenization and stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset and ensure text column is string type
df = pd.read_csv('TweetSentiment.csv', encoding='ISO-8859-1')
df['text'] = df['text'].fillna('').astype(str)

# Map sentiment labels to numeric values
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['sentiment'].map(label_map)

# Task 1: Exploratory Data Analysis
# 1. Plot sentiment distribution as a bar chart
df['sentiment'].value_counts().plot(kind='bar', figsize=(6, 4))
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda_sentiment_distribution.png', dpi=300)
plt.close()

# 2. Plot distribution of tweet lengths
df['text_length'] = df['text'].str.len()
plt.figure(figsize=(6, 4))
sns.histplot(df['text_length'], bins=30)
plt.title('Tweet Length Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('eda_length_distribution.png', dpi=300)
plt.close()

# 3. Generate word cloud for all tweets
all_text = ' '.join(df['text'])
wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=set(stopwords.words('english'))
)
wc.generate(all_text)
wc.to_file('eda_wordcloud.png')

# 4. Boxplot of word count distribution by sentiment
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 6))
sns.boxplot(x='sentiment', y='word_count', data=df)
plt.title('Word Count Distribution by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Word Count')
plt.tight_layout()
plt.savefig('eda_wordcount_boxplot.png', dpi=300)
plt.close()

# Task 2: Pre-processing
# Set up engine to convert numeric tokens to words, stopword list, and stemmer
number_engine = inflect.engine()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Function to convert digits into their word representation
def convert_numbers_to_words(text: str) -> str:
    return re.sub(r"\b\d+\b", lambda m: number_engine.number_to_words(m.group()), text)

# Clean text by removing handles, hashtags, URLs, extra whitespace, and converting to lowercase
def clean_text(text: str) -> str:
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return convert_numbers_to_words(text)

# Tokenize, remove non-alphabetic tokens, apply stemming, and drop stopwords
def preprocess_tokens(text: str) -> str:
    tokens = nltk.word_tokenize(text)
    cleaned = []
    for tok in tokens:
        if tok.isalpha() and tok not in stop_words:
            cleaned.append(stemmer.stem(tok))
    return ' '.join(cleaned)

# Apply cleaning and tokenization to the DataFrame
df['clean_text'] = df['text'].apply(clean_text)
df['processed'] = df['clean_text'].apply(preprocess_tokens)

# --- Task 3: Text Classification (NB & MLP) ---
# Split data into training and test sets
X = df['processed']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define various count and TF-IDF vectorizers
vectorizers = [
    ('Count_Unigram', CountVectorizer(ngram_range=(1, 1))),
    ('Count_UnigramBigram', CountVectorizer(ngram_range=(1, 2))),
    ('Tfidf_Unigram', TfidfVectorizer(ngram_range=(1, 1))),
    ('Tfidf_UnigramBigram', TfidfVectorizer(ngram_range=(1, 2)))
]

# Factory to create preprocessing transformers with optional stopword removal and stemming
def make_preprocessor(remove_stopwords=False, do_stemming=False):
    class TextPreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.remove_stopwords = remove_stopwords
            self.do_stemming = do_stemming
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return X.apply(self._clean)

        def _clean(self, text):
            text = text.lower()
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
            text = re.sub(r"[^a-z0-9\s]", "", text)
            tokens = text.split()
            if self.remove_stopwords:
                tokens = [w for w in tokens if w not in self.stop_words]
            if self.do_stemming:
                tokens = [self.stemmer.stem(w) for w in tokens]
            return " ".join(tokens)

    return TextPreprocessor()

# Different preprocessing configurations to test
preprocessing_sets = [
    ('none', make_preprocessor(False, False)),
    ('stop', make_preprocessor(True, False)),
    ('stem', make_preprocessor(False, True)),
    ('stop_stem', make_preprocessor(True, True))
]

# Define Naive Bayes and MLP classifiers
classifiers = [
    ('NB', MultinomialNB()),
    ('MLP', MLPClassifier(
        hidden_layer_sizes=(50,),
        activation='relu',
        solver='adam',
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42
    ))
]

# Transformer to convert sparse matrix to dense for MLP
to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)

# Evaluate each combination of vectorizer, preprocessor, and classifier
results = []
for vec_name, vec in vectorizers:
    for prep_name, prep in preprocessing_sets:
        for clf_name, clf in classifiers:
            steps = [('clean', prep), ('vect', vec)]
            if clf_name == 'MLP':
                steps += [('scale', MaxAbsScaler()), ('to_dense', to_dense), ('clf', clf)]
            else:
                steps += [('clf', clf)]

            pipe = Pipeline(steps)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            print(f"{clf_name} | {vec_name} | {prep_name} -> Acc: {acc:.4f}, F1: {f1:.4f}")
            results.append({
                'clf': clf_name,
                'vec': vec_name,
                'prep': prep_name,
                'accuracy': acc,
                'f1_macro': f1
            })

# Display performance summary as a pivot table
pd.DataFrame(results).pivot_table(
    index=['clf', 'vec'],
    columns='prep',
    values=['accuracy', 'f1_macro']
)

# Task 4: Semantic Similarity with BERT
# Load tokenizer and base model for embedding extraction
tok = BertTokenizer.from_pretrained('bert-base-uncased')
mod = BertModel.from_pretrained('bert-base-uncased')
mod.eval()

# Function to extract [CLS] embedding for a given text
def get_embed(txt: str):
    inp = tok(txt, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        out = mod(**inp)
    return out.last_hidden_state[0, 0].cpu().numpy()

# Sample positive tweets and compute pairwise cosine similarity
samples = df[df['label'] == 2]['processed'].sample(16, random_state=42).tolist()
embs = [get_embed(s) for s in samples]
for i in range(15):
    s1, s2 = samples[i], samples[i + 1]
    sim = cosine_similarity(embs[i].reshape(1, -1), embs[i + 1].reshape(1, -1))[0, 0]
    print(f"Pair {i+1}:")
    print(f"  Sentence 1: {s1}")
    print(f"  Sentence 2: {s2}")
    print(f"  Cosine similarity: {sim:.4f}\n")

# Task 5: BERT Fine-Tuning with Subsampling
# Subsample dataset for faster fine-tuning experiments
sample_df = df.sample(n=5000, random_state=42).reset_index(drop=True)
hf_df = sample_df[['processed', 'sentiment']].rename(columns={'processed': 'text'})
hf_df['label'] = hf_df['sentiment'].map(label_map)

# Split into training and validation sets and convert to Hugging Face Datasets
train_hf, valid_hf = train_test_split(
    hf_df[['text', 'label']], test_size=0.2, stratify=hf_df['label'], random_state=42
)
train_ds = Dataset.from_pandas(train_hf.reset_index(drop=True))
valid_ds = Dataset.from_pandas(valid_hf.reset_index(drop=True))

# Tokenize and format datasets for PyTorch
train_ds = train_ds.map(
    lambda batch: tok(batch['text'], padding='max_length', truncation=True, max_length=128),
    batched=True
)
valid_ds = valid_ds.map(
    lambda batch: tok(batch['text'], padding='max_length', truncation=True, max_length=128),
    batched=True
)
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
valid_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Initialize and fine-tune a classification head on top of BERT
model_mt = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=3
)
args = TrainingArguments(
    output_dir='./bert_subset',
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01
)
trainer = Trainer(
    model=model_mt,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tok,
    compute_metrics=lambda pred: {
        'acc': accuracy_score(pred.label_ids, np.argmax(pred.predictions, axis=1))
    }
)

# Train and evaluate the fine-tuned model
trainer.train()
res = trainer.evaluate()
print(res)
