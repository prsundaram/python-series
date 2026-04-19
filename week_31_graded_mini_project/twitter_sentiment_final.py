# Twitter Sentiment Analysis - Final Pipeline
# Baseline → Hyperparameter Tuning → Transfer Learning

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization

from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('data/twitter_training.csv', header=None)
df.columns = ['id', 'entity', 'sentiment', 'text']

# Cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Tokenization
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['clean_text'])

X = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(X, maxlen=max_len)

y = pd.get_dummies(df['sentiment']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Baseline Model
def build_baseline():
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(LSTM(64))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

baseline_model = build_baseline()
baseline_model.fit(X_train, y_train, epochs=3, validation_split=0.2)

y_pred = np.argmax(baseline_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

baseline_acc = accuracy_score(y_true, y_pred)
print("Baseline Accuracy:", baseline_acc)

# Hyperparameter Tuning
def build_tuned(units=64, dropout=0.5):
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(LSTM(units))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

best_acc = 0
best_params = None

for units in [32, 64]:
    for dropout in [0.3, 0.5]:
        model = build_tuned(units, dropout)
        history = model.fit(X_train, y_train, epochs=3, validation_split=0.2, verbose=0)
        val_acc = max(history.history['val_accuracy'])
        if val_acc > best_acc:
            best_acc = val_acc
            best_params = (units, dropout)

print("Best Params:", best_params)

tuned_model = build_tuned(best_params[0], best_params[1])
tuned_model.fit(X_train, y_train, epochs=5, validation_split=0.2)

y_pred = np.argmax(tuned_model.predict(X_test), axis=1)
tuned_acc = accuracy_score(y_true, y_pred)
print("Tuned Accuracy:", tuned_acc)

# Transfer Learning (GloVe)
embedding_dim = 100
embeddings_index = {}

try:
    with open('glove.twitter.27B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

    embedding_matrix = np.zeros((max_words, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if i < max_words:
            vec = embeddings_index.get(word)
            if vec is not None:
                embedding_matrix[i] = vec

    def build_transfer():
        model = Sequential()
        model.add(Embedding(max_words, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False))
        model.add(LSTM(best_params[0]))
        model.add(Dropout(best_params[1]))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    transfer_model = build_transfer()
    transfer_model.fit(X_train, y_train, epochs=5, validation_split=0.2)

    y_pred = np.argmax(transfer_model.predict(X_test), axis=1)
    transfer_acc = accuracy_score(y_true, y_pred)

    print("Transfer Learning Accuracy:", transfer_acc)

except:
    print("GloVe file not found, skipping transfer learning")

print("\nFinal Comparison:")
print("Baseline:", baseline_acc)
print("Tuned:", tuned_acc)
