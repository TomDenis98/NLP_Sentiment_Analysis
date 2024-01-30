import os
import nltk
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Download NLTK resources if not already downloaded
#nltk.download('stopwords')
#nltk.download('punkt')

print('test')

def read_files(directory):
    corpus = []
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), "r") as text:
            corpus.append(text.read())
    return corpus

reviews_directory = "C:\\Users\\HP\\OneDrive\\Documenten\\TERM 2\\NLP\\Assignment\\Data"
corpus = read_files(reviews_directory)

# Other functions remain unchanged

# Preprocessing function
def preprocessing(raw_text):
    stop_words = set(stopwords.words('english'))
    processed_text = []
    for source in raw_text:
        lines = source.lower().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                text = ' '.join(parts[:-1])
                words = word_tokenize(text)

                filtered_words = [word for word in words if word not in stop_words]
                filtered_text = ' '.join(filtered_words)

                try:
                    label = int(parts[-1])
                    processed_text.append((filtered_text, label))
                except ValueError:
                    pass  # Handle invalid labels here

    return processed_text

# Preprocess data
processed_db = preprocessing(corpus)
X, y = zip(*processed_db)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# BAG OF WORDS CLASSIFICATION
vectorizer_BOW = CountVectorizer()
X_train_vectorized_BOW = vectorizer_BOW.fit_transform(X_train)
X_test_vectorized_BOW = vectorizer_BOW.transform(X_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized_BOW, y_train)

y_pred_BOW = nb_classifier.predict(X_test_vectorized_BOW)

print("Classification Report for BOW:")
print(classification_report(y_test, y_pred_BOW))

# TDIDF CLASSIFICATION
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

y_pred = nb_classifier.predict(X_test_vectorized)

print("Classification Report for TF-IDF:")
print(classification_report(y_test, y_pred))

# N-GRAMS CLASSIFICATION
vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

y_pred = nb_classifier.predict(X_test_vectorized)

print("Classification Report for N-grams:")
print(classification_report(y_test, y_pred))
