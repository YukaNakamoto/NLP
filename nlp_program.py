import pandas as pd  # Data handling
import numpy as np  # Numerical operations
import re  # Regular expressions for text cleaning
import inflect  # Convert numbers to words
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Statistical data visualization
from wordcloud import WordCloud, STOPWORDS  # Word cloud generation
from sklearn.model_selection import train_test_split  # Train/test split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # Feature extraction
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier
from sklearn.metrics import accuracy_score, f1_score, classification_report  # Evaluation metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler, FunctionTransformer
import nltk  # Natural Language Toolkit
from nltk.corpus import stopwords  # Stopword list
from nltk.stem import PorterStemmer
import torch  # PyTorch for embeddings and models
from transformers import (  # Hugging Face Transformers
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset  # Hugging Face Dataset
from sklearn.metrics.pairwise import cosine_similarity  # Similarity computation

# --- Setup & Data Loading ---
# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Load the CSV dataset and fill missing texts
df = pd.read_csv('TweetSentiment.csv', encoding='ISO-8859-1')
df['text'] = df['text'].fillna('').astype(str)
# Encode sentiment labels to numeric for classifier compatibility
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])

# --- Preprocessing Utilities ---
number_engine = inflect.engine()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def convert_numbers_to_words(text: str) -> str:
    return re.sub(r"\b\d+\b", lambda m: number_engine.number_to_words(m.group()), text)

def clean_text(text: str) -> str:
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = text.lower().strip()
    return re.sub(r"\s+", " ", convert_numbers_to_words(text))

def preprocess_tokens(text: str) -> str:
    tokens = nltk.word_tokenize(text)
    return ' '.join(
        stemmer.stem(tok)
        for tok in tokens
        if tok.isalpha() and tok not in stop_words
    )

df['clean_text'] = df['text'].apply(clean_text)
df['processed'] = df['clean_text'].apply(preprocess_tokens)

# --- Task 3: Vectorization & Model Comparison (NB & MLP) ---
X = df['processed']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorizers: unigram and unigram+bigram for Count & TF-IDF
vectorizers = [
    ('Count_Unigram', CountVectorizer(ngram_range=(1,1))),
    ('Count_UnigramBigram', CountVectorizer(ngram_range=(1,2))),
    ('Tfidf_Unigram', TfidfVectorizer(ngram_range=(1,1))),
    ('Tfidf_UnigramBigram', TfidfVectorizer(ngram_range=(1,2)))
]

# Preprocessing pipelines
def make_preprocessor(remove_stopwords=False, do_stemming=False):
    class TextPreprocessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.remove_stopwords = remove_stopwords
            self.do_stemming = do_stemming
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        def fit(self, X, y=None): return self
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

preprocessing_sets = [
    ('none', make_preprocessor(False, False)),
    ('stop', make_preprocessor(True, False)),
    ('stem', make_preprocessor(False, True)),
    ('stop_stem', make_preprocessor(True, True))
]

# Classifiers
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

# To dense transformer for MLP
to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)

# Evaluate models
results = []
for vec_name, vec in vectorizers:
    for prep_name, prep in preprocessing_sets:
        for clf_name, clf in classifiers:
            steps = [('clean', prep), ('vect', vec)]
            if clf_name == 'MLP':
                steps.extend([
                    ('scale', MaxAbsScaler()),
                    ('to_dense', to_dense),
                    ('clf', clf)
                ])
            else:
                steps.append(('clf', clf))
            pipe = Pipeline(steps)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            print(f"{clf_name} | {vec_name} | {prep_name} -> Acc: {acc:.4f}, F1-macro: {f1:.4f}")
            results.append({
                'classifier': clf_name,
                'vectorizer': vec_name,
                'preprocessing': prep_name,
                'accuracy': acc,
                'f1_macro': f1
            })

# Summarize results
results_df = pd.DataFrame(results)
print("\nSummary of configurations:")
print(results_df.pivot_table(
    index=['classifier','vectorizer'],
    columns='preprocessing',
    values=['accuracy','f1_macro']
))

# --- Task 4: Semantic Similarity with BERT ---
tok = BertTokenizer.from_pretrained('bert-base-uncased')
mod = BertModel.from_pretrained('bert-base-uncased')
mod.eval()

def get_embed(txt: str):
    inp = tok(txt, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad(): out = mod(**inp)
    return out.last_hidden_state[0,0].cpu().numpy()

samples = df[df['sentiment']=='positive']['processed'].sample(15, random_state=42)
embs = [get_embed(s) for s in samples]
print("Semantic Similarities:")
for i in range(5):
    sim = cosine_similarity(embs[i].reshape(1,-1), embs[i+1].reshape(1,-1))[0,0]
    print(f"Pair {i+1}: {sim:.4f}")

# --- Task 5: BERT Fine-Tuning with Subsampling ---
SAMPLE_SIZE = 5000  # Number of examples to sample for faster training
sample_df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
hf_df = sample_df[['processed', 'sentiment']].rename(columns={'processed': 'text'})
hf_df['label'] = hf_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
train_hf, valid_hf = train_test_split(
    hf_df[['text', 'label']], test_size=0.2, stratify=hf_df['label'], random_state=42
)
train_ds = Dataset.from_pandas(train_hf.reset_index(drop=True))
valid_ds = Dataset.from_pandas(valid_hf.reset_index(drop=True))

def tok_map(batch):
    return tok(batch['text'], padding='max_length', truncation=True, max_length=128)

train_ds = train_ds.map(tok_map, batched=True)
valid_ds = valid_ds.map(tok_map, batched=True)
train_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
valid_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

model_mt = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
args = TrainingArguments(
    output_dir='./bert_sentiment_subset',
    do_train=True,
    do_eval=True,
    logging_dir='./logs_subset',
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch'
)

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average='weighted')
    }

trainer = Trainer(
    model=model_mt,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tok,
    compute_metrics=compute_metrics
)

print("=== Fine-tuning BERT on Sampled Subset ===")
trainer.train()
res = trainer.evaluate()
print("BERT Fine-tuning Results on Subset:", res)
