import pandas as pd  # Data handling
import numpy as np  # Numerical operations
import re  # Regular expressions for text cleaning
import inflect  # Convert numbers to words
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Statistical data visualization
from wordcloud import WordCloud  # Word cloud generation
from sklearn.model_selection import train_test_split  # Train/test split
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF feature extraction
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score  # Evaluation metrics
from sklearn.utils import resample  # Resampling for class balancing
import nltk  # Natural Language Toolkit
from nltk.corpus import stopwords  # Stopword list
from nltk.stem import PorterStemmer  # Stemming
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
from tensorflow.keras import Sequential  # Keras Sequential API
from tensorflow.keras.layers import Dense, Dropout  # Dense and Dropout layers

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the CSV dataset into a pandas DataFrame
df = pd.read_csv('TweetSentiment.csv', encoding='ISO-8859-1')

# --- Task 1: Exploratory Data Analysis ---
# Plot distribution of sentiment labels
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('eda_sentiment_distribution.png', dpi=300)
plt.close()

# Compute and plot tweet length distribution
df['text_length'] = df['text'].str.len()
plt.figure(figsize=(6,4))
sns.histplot(df['text_length'], bins=30)
plt.title('Tweet Length Distribution')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.savefig('eda_length_distribution.png', dpi=300)
plt.close()

# Generate a word cloud of all tweets
all_text = ' '.join(df['text'].fillna('').astype(str))
wc = WordCloud(
    width=800,
    height=400,
    background_color='white',
    stopwords=set(stopwords.words('english'))
)
wc.generate(all_text)
wc.to_file('eda_wordcloud.png')

# --- Task 2: Preprocessing ---
# Initialize number-to-word engine, stopwords set, and stemmer
number_engine = inflect.engine()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def convert_numbers_to_words(text: str) -> str:
    """
    Replace standalone digits with their English word equivalents.
    E.g., "I have 2 dogs" -> "I have two dogs"
    """
    def replace(match):
        return number_engine.number_to_words(match.group())
    return re.sub(r"\b\d+\b", replace, text)

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
      - Remove mentions (@username) and hashtags (#tag)
      - Strip URLs
      - Lowercase the text and collapse extra spaces
      - Convert numbers to words
    """
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return convert_numbers_to_words(text)

def preprocess_tokens(text: str) -> str:
    """
    Tokenize cleaned text, filter out non-alphabetic tokens and stopwords,
    then apply Porter stemming.
    """
    tokens = nltk.word_tokenize(text)
    filtered = [
        stemmer.stem(tok)
        for tok in tokens
        if tok.isalpha() and tok not in stop_words
    ]
    return ' '.join(filtered)

# Apply cleaning and tokenization to the dataset
df['clean_text'] = df['text'].fillna('').astype(str).apply(clean_text)
df['processed']  = df['clean_text'].apply(preprocess_tokens)

# --- Task 3: Modeling ---
X = df['processed']
y = df['sentiment']

# Split into train/test sets and then balance the training set by oversampling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
train_df  = pd.concat([X_train, y_train], axis=1)
max_count = train_df['sentiment'].value_counts().max()
# Oversample minority classes to match the majority class count
over = [
    grp.sample(max_count, replace=True, random_state=42)
    for _, grp in train_df.groupby('sentiment')
]
train_bal = pd.concat(over)
X_train   = train_bal['processed']
y_train   = train_bal['sentiment']

# 3.1 Multinomial Naive Bayes
vect_nb = TfidfVectorizer(max_features=5000)  # Limit vocabulary size
X_tr_nb = vect_nb.fit_transform(X_train)
X_te_nb = vect_nb.transform(X_test)
bayes   = MultinomialNB().fit(X_tr_nb, y_train)
y_nb    = bayes.predict(X_te_nb)
print("Naive Bayes Report:\n", classification_report(y_test, y_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_nb))

# 3.2 Feed-Forward Neural Network
vect_nn   = TfidfVectorizer(max_features=5000)
X_tr_nn   = vect_nn.fit_transform(X_train).toarray()
X_te_nn   = vect_nn.transform(X_test).toarray()
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_tr_nn.shape[1],)),  # Hidden layer
    Dropout(0.5),  # Prevent overfitting
    Dense(3, activation='softmax')  # Output layer for 3 classes
])
nn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Train the FFNN with a 10% validation split
nn_model.fit(
    X_tr_nn,
    y_train.map(label_map),
    epochs=5,
    batch_size=32,
    validation_split=0.1
)
y_nn = np.argmax(nn_model.predict(X_te_nn), axis=1)
print("FFNN Report:\n", classification_report(y_test.map(label_map), y_nn))
print("Confusion Matrix:\n", confusion_matrix(y_test.map(label_map), y_nn))

# 3.3 Binary Classification (negative vs. positive)
bin_df = df[df['sentiment'] != 'neutral']  # Drop neutral examples
Xb, yb = bin_df['processed'], bin_df['sentiment']
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    Xb, yb, test_size=0.2, stratify=yb, random_state=42
)
vect_bin = TfidfVectorizer(max_features=3000)
Xb_tr_v  = vect_bin.fit_transform(Xb_tr)
Xb_te_v  = vect_bin.transform(Xb_te)
bnb      = MultinomialNB().fit(Xb_tr_v, yb_tr)
y_bnb    = bnb.predict(Xb_te_v)
print("Binary NB Report:\n", classification_report(yb_te, y_bnb))

# --- Task 4: Semantic Text Similarity with BERT ---
tok = BertTokenizer.from_pretrained('bert-base-uncased')  # Tokenizer
mod = BertModel.from_pretrained('bert-base-uncased')  # Pretrained model
mod.eval()  # Set to evaluation mode

def get_embed(txt: str):
    """
    Obtain the [CLS] token embedding for a single text via BERT.
    """
    inp = tok(
        txt,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=128
    )
    with torch.no_grad():
        out = mod(**inp)
    return out.last_hidden_state[0, 0].cpu().numpy()

# Sample a few positive tweets to compute pairwise cosine similarities
samples = df[df['sentiment'] == 'positive']['processed'].sample(15, random_state=42)
embs    = [get_embed(s) for s in samples]
print("Semantic Similarities:")
for i in range(5):
    sim = cosine_similarity(
        embs[i].reshape(1, -1),
        embs[i+1].reshape(1, -1)
    )[0, 0]
    print(f"Pair {i+1}: {sim:.4f}")

# --- Task 5: BERT Fine-Tuning with Subsampling ---
SAMPLE_SIZE = 5000  # Number of examples to sample for faster training

# Randomly sample from the full dataset
sample_df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# Prepare DataFrame for Hugging Face Trainer
hf_df = sample_df[['processed', 'sentiment']].rename(columns={'processed': 'text'})
hf_df['label'] = hf_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

# Split into train and validation sets
train_hf, valid_hf = train_test_split(
    hf_df[['text', 'label']],
    test_size=0.2,
    stratify=hf_df['label'],
    random_state=42
)

# Convert pandas DataFrames into Hugging Face Datasets
train_ds = Dataset.from_pandas(train_hf.reset_index(drop=True))
valid_ds = Dataset.from_pandas(valid_hf.reset_index(drop=True))

def tok_map(batch):
    """
    Tokenize a batch of texts for BERT fine-tuning.
    """
    return tok(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

# Apply tokenization to the datasets
train_ds = train_ds.map(tok_map, batched=True)
valid_ds = valid_ds.map(tok_map, batched=True)
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
valid_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load a pre-trained BERT model for sequence classification
model_mt = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
)

# Define training arguments
args = TrainingArguments(
    output_dir='./bert_sentiment_subset',  # Where to save checkpoints
    do_train=True,
    do_eval=True,
    logging_dir='./logs_subset',  # TensorBoard logs
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='epoch'
)

def compute_metrics(pred):
    """
    Compute accuracy and weighted F1 score for evaluation during training.
    """
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average='weighted')
    }

# Initialize the Trainer and run fine-tuning
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