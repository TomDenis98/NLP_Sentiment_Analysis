import os
import string
import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords, PlaintextCorpusReader
from nltk.tokenize import word_tokenize
from scipy.sparse import issparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# preprocessing
def read_text_files_in_directory(directory_path):
    # Assuming each text file in the directory is considered a separate document
    corpus_reader = PlaintextCorpusReader(directory_path, '.*\.txt')

    # Read all lines from all documents
    all_lines = corpus_reader.raw().splitlines()

    return all_lines


def preprocess_text(sentences):
    """
    Preprocesses a list of sentences by performing the following steps:

    1. Extracts the sentence part before '\t' (tab character).
    2. Removes punctuation from the extracted text.
    3. Removes stop words and tokenizes the text.
    4. Filters out stop words from the tokens.
    5. Joins the filtered tokens into a processed sentence.

    Parameters:
    - sentences (list): A list of sentences to be preprocessed.

    Returns:
    - list: A list of processed sentences after the specified preprocessing steps.
    """
    processed_sentences = []

    for sentence in sentences:
        # Extracting the sentence part before '\t' (tab character)
        text = sentence.split('\t')[0]

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove stop words and tokenize
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word not in stop_words]

        processed_sentence = ' '.join(filtered_tokens)  # Join the tokens into a String
        processed_sentences.append(processed_sentence)

    return processed_sentences


def visualize_total_words_per_sentence_distribution(directory_path):
    # Get all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    for file_name in files:
        file_path = os.path.join(directory_path, file_name)

        # Read sentences from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                sentences = file.readlines()
            except UnicodeDecodeError:
                print(f"Error reading file: {file_path}")
                continue

        # Calculate the total words per sentence
        total_words_per_sentence = [len(sentence.split()) for sentence in sentences]

        # Plotting the histogram for total words per sentence
        plt.figure(figsize=(10, 6))
        plt.hist(total_words_per_sentence, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Total Words per Sentence')
        plt.ylabel('Frequency')
        plt.title(f'Total Words per Sentence Distribution - {file_name}')
        plt.show()

# prediction model
def naive_bayes_classification(df, train_column, goal_column, threshold=0.5):
    """
    Perform Naive Bayes binary classification on a DataFrame containing sentences and sentiments.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing 'sentence' and 'sentiment' columns.
    - threshold (float): Threshold for binary classification. Default is 0.5.

    Returns:
    - tuple: A tuple containing the accuracy and classification report.
    """
    # Vectorize the sentences to convert them into numerical features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[train_column])

    # Extract labels
    y = df[goal_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    # Predict the labels on the test set
    y_pred = nb_classifier.predict(X_test)

    # Evaluate the classifier
    accuracy_bayes = accuracy_score(y_test, y_pred)
    classification_rep_bayes = classification_report(y_test, y_pred)

    return accuracy_bayes, classification_rep_bayes


# experiment 1
def lemmatize(df, column):
    """
    Lemmatizes the words in a specified column of the DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the specified column.
    - column (str): The column to lemmatize.

    Returns:
    - pandas.DataFrame: DataFrame with lemmatized words in the specified column.
    """

    # Create a WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize and stem each word in each sentence in the specified column
    df[column] = df[column].apply(lambda sent: ' '.join(
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tag(word_tokenize(sent))))

    return df


def get_wordnet_pos(pos_tag):
    """
       Maps POS (Part of Speech) tags to WordNet POS tags.

       Parameters:
       - pos_tag (str): POS tag obtained from NLTK.

       Returns:
       - str: WordNet POS tag.

       Examples:
       >>> get_wordnet_pos('JJ')
       'a'
       >>> get_wordnet_pos('VB')
       'v'
       >>> get_wordnet_pos('NN')
       'n'
       >>> get_wordnet_pos('RB')
       'r'
       >>> get_wordnet_pos('XYZ')  # Default to 'n' for unknown tags
       'n'
       """
    if pos_tag.startswith('J'):
        return 'a'  # adjective
    elif pos_tag.startswith('V'):
        return 'v'  # verb
    elif pos_tag.startswith('N'):
        return 'n'  # noun
    elif pos_tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # default to noun if POS tag is not found


# Experiment 2
def apply_tfidf_and_classify(df, text_column, sentiment_column):
    """
    Apply TF-IDF to the input DataFrame and perform Naive Bayes classification on the sentiment column.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing 'text_column' and 'sentiment_column'.
    - text_column (str): Name of the column with text data.
    - sentiment_column (str): Name of the column with sentiment labels.

    Prints:
    - Displays the first 5 rows of the DataFrame after applying TF-IDF.
    - Prints accuracy and classification report.

    Returns:
    - None
    """
    # Apply TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(df[text_column])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df[sentiment_column], test_size=0.2, random_state=42)

    # Train a Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    # Predict the labels on the test set
    y_pred = nb_classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Replace the 'sentences_preprocessed' column with the TF-IDF-transformed data
    df[text_column] = pd.Series(list(X_tfidf))

    # Display the first 5 rows of the DataFrame after applying TF-IDF
    print("First 5 rows of the DataFrame after applying TF-IDF:")
    print(df.head())

    # Print accuracy and classification report
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_rep)


# experiment 3
def apply_ngram_and_classify(df, text_column, sentiment_column, ngram_range=(1, 2)):
    """
    Apply N-grams to the input DataFrame and perform Naive Bayes classification on the sentiment column.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing 'text_column' and 'sentiment_column'.
    - text_column (str): Name of the column with text data.
    - sentiment_column (str): Name of the column with sentiment labels.
    - ngram_range (tuple): Range for N-grams. Default is unigrams and bigrams (1, 2).

    Prints:
    - Displays the first 5 rows of the DataFrame after applying N-grams.
    - Prints accuracy and classification report.

    Returns:
    - None
    """
    # Check if the input data is already a sparse matrix
    is_sparse = issparse(df[text_column])

    # Apply N-grams
    ngram_vectorizer = CountVectorizer(ngram_range=ngram_range, lowercase=not is_sparse)
    X_ngrams = ngram_vectorizer.fit_transform(df[text_column])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_ngrams, df[sentiment_column], test_size=0.2, random_state=42)

    # Train a Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    # Predict the labels on the test set
    y_pred = nb_classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Replace the 'text_column' column with the N-grams-transformed data
    df[text_column] = pd.Series(list(X_ngrams))

    # Display the first 5 rows of the DataFrame after applying N-grams
    print("First 5 rows of the DataFrame after applying N-grams:")
    print(df.head())

    # Print accuracy and classification report
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_rep)


# experiment 4
def bow_and_classify(df, text_column, sentiment_column):
    """
    Apply Bag of Words (BOW) to the input DataFrame and perform Naive Bayes classification on the sentiment column.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing 'text_column' and 'sentiment_column'.
    - text_column (str): Name of the column with text data.
    - sentiment_column (str): Name of the column with sentiment labels.

    Prints:
    - Displays the first 5 rows of the DataFrame after applying BOW.
    - Prints accuracy and classification report.

    Returns:
    - None
    """
    # Apply Bag of Words
    bow_vectorizer = CountVectorizer()
    X_bow = bow_vectorizer.fit_transform(df[text_column])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_bow, df[sentiment_column], test_size=0.2, random_state=42)

    # Train a Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    # Predict the labels on the test set
    y_pred = nb_classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Replace the 'text_column' column with the BOW-transformed data
    df[text_column] = pd.Series(list(X_bow))

    # Display the first 5 rows of the DataFrame after applying BOW
    print("First 5 rows of the DataFrame after applying BOW:")
    print(df.head())

    # Print accuracy and classification report
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_rep)


def glove_sentiment_analysis(df, glove_model_path):
    """
    Applies sentiment analysis using pre-trained GloVe word vectors.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing preprocessed sentences and sentiments.
    - glove_model_path (str): Path to the pre-trained GloVe model file.

    Returns:
    None

    This function performs the following steps:
    1. Loads a pre-trained GloVe word vector model.
    2. Applies the model to obtain average GloVe vectors for each preprocessed sentence.
    3. Ensures non-negativity in GloVe vectors.
    4. Splits the data into training and testing sets.
    5. Trains a Naive Bayes classifier on the training set.
    6. Makes predictions on the test set.
    7. Prints the first 5 rows of the DataFrame after applying GloVe.
    8. Prints accuracy and classification report based on Naive Bayes predictions.
    """
    # Load pre-trained GloVe model
    glove_model = load_glove_model(glove_model_path)

    # Apply the function to obtain average GloVe vectors for each sentence
    df['glove_vectors'] = df['sentences_preprocessed'].apply(
        lambda sentence: get_avg_glove_vector(sentence, glove_model)
    )

    # Ensure non-negativity in GloVe vectors
    df['glove_vectors'] = df['glove_vectors'].apply(lambda vec: vec - np.min(vec))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        np.vstack(df['glove_vectors']), df['sentiments'], test_size=0.2, random_state=42
    )

    # Train the Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)
    # Print head of the DataFrame after applying GloVe
    print("\nFirst 5 rows of the DataFrame after applying GLOVE:")
    print(df.head())

    # Print accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy after appying GLOVE: {accuracy:.2f}")

    class_report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(class_report)


def load_glove_model(file_path):
    """
    Load GloVe model from a file.

    Parameters:
    - file_path (str): Path to the GloVe model file.

    Returns:
    - dict: A dictionary where keys are words, and values are corresponding GloVe vectors.
    """
    model = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(val) for val in parts[1:]])
            model[word] = vector
    return model


def get_avg_glove_vector(sentence, glove_model):
    """
    Calculate the average GloVe vector for a sentence.

    Parameters:
    - sentence (str): Input sentence.
    - glove_model (dict): GloVe model loaded as a dictionary.

    Returns:
    - numpy.ndarray: Average GloVe vector for the input sentence.
    """
    vectors = [glove_model[word] for word in sentence.split() if word in glove_model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(len(next(iter(glove_model.values()))))


# Experiment 5
def apply_lda_and_sentiment_analysis(df, text_column, sentiment_column, num_topics=5):
    """
    Apply Latent Dirichlet Allocation (LDA) and sentiment analysis to the input DataFrame.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing 'text_column' and 'sentiment_column'.
    - text_column (str): Name of the column with text data.
    - sentiment_column (str): Name of the column with sentiment labels.
    - num_topics (int): Number of topics for LDA. Default is 5.

    Prints:
    - Displays the topics and associated words.
    - Displays the distribution of topics for the first 5 documents.
    - Prints accuracy and classification report for sentiment analysis.

    Returns:
    - None
    """
    # Create a CountVectorizer to convert the text data to a bag-of-words representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[text_column])

    # Apply LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Display topics and associated words
    feature_names = vectorizer.get_feature_names_out()
    print("Top words for each topic:")
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

    # Display the distribution of topics for the first 5 documents
    doc_topics = lda.transform(X[:5])
    print("\nTopic distribution for the first 5 documents:")
    print(pd.DataFrame(doc_topics, columns=[f"Topic {i}" for i in range(1, num_topics + 1)]))

    # Split the data into training and testing sets for sentiment analysis
    X_train, X_test, y_train, y_test = train_test_split(X, df[sentiment_column], test_size=0.2, random_state=42)

    # Train a Naive Bayes classifier for sentiment analysis
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    # Predict the labels on the test set
    y_pred = nb_classifier.predict(X_test)

    # Evaluate the classifier for sentiment analysis
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Print accuracy and classification report for sentiment analysis
    print("\nSentiment Analysis Results:")
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_rep)
