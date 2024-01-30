from scipy.stats import f_oneway
from Functions import *

nltk.download('wordnet')
pd.set_option('display.max_rows', 100)  # Show all rows
pd.set_option('display.max_columns', 100)  # Show all columns
pd.set_option('display.max_colwidth', None)

# Get the directory path of the current Python file
current_directory = os.path.dirname(__file__)

# Construct the necessary paths
root_folder = current_directory.replace("Code", "Data")
glove_folder = current_directory.replace("Code", "Glove\\glove.6B.50d.txt")

# Initialize variables
num_documents = 0
total_word_count = 0
unique_words = set()
word_counts_per_document = []

print('Data_Preprocessing')
sentences = []
sentiments = []
# Error sentences are saved separately so investigation of these cases can be done
error_sentences = []
# Read text files in directory and store as separate rows
rows = read_text_files_in_directory(root_folder)

# select each individual sentence in rows to split the sentiment and sentence from themselves
for row in rows:
    if len(row) > 0:  # Check if the row is not an empty string
        if row[-1] in ['0', '1']:  # Check if the last character is a valid sentiment ('0' or '1')
            sentence = row[:-1]  # Extract the sentence without the last character
            sentiment = int(row[-1])  # Extract the last character as the sentiment and convert it to an integer
            sentences.append(sentence)
            sentiments.append(sentiment)
        else:
            # If the last character is not a valid sentiment, add to error_sentences
            error_sentences.append(row)

# change sentiment from String to Integer
sentiments = [int(sentiment) for sentiment in sentiments]
# Preprocess the sentences list using the preprocess_text function
preprocessed_sentences = preprocess_text(sentences)

# testing
print('raw sentences:', sentences[0:3])
print('sentiments:', sentiments[0:3])
print('preprocessed sentences:', preprocessed_sentences[0:3])
print('sentences that could not be split:', error_sentences[0:3])  # Display the first 3 error sentences

# Create a DataFrame from the preprocessed sentences and sentiments
rawdf = pd.DataFrame({'sentences_raw': sentences, 'sentiments': sentiments})
df = pd.DataFrame({'sentences_preprocessed': preprocessed_sentences, 'sentiments': sentiments})
print('raw dataset:\n', rawdf[:3])
print('preprocessed dataset:\n', df[:3])

print('Experiment Set-Up')
# Perform Naive Bayes classification to create a baseline for the experiments
accuracy_baseline, classification_rep_baseline = naive_bayes_classification(df, 'sentences_preprocessed', 'sentiments')
print("The bayes baseline accuracy before the experiments is:\n", accuracy_baseline)
print("Classification Report Naive Bayes:")
print(classification_rep_baseline)

print('Experiments 1 : Effect of lemmatization on Naive Bayes')
df_lemmatized = lemmatize(df.copy(), 'sentences_preprocessed')
print('lemmatized:\n', df_lemmatized[:5])
accuracy_lemmatization, classification_lemmatization = naive_bayes_classification(df_lemmatized,'sentences_preprocessed','sentiments')
print("Experiment 1: Naive Bayes Accuracy With lemmatization\n", accuracy_lemmatization)
print("Classification Report:\n", classification_lemmatization)

print('Experiment 2: Effect of TF.IDF weights')
apply_tfidf_and_classify(df.copy(), 'sentences_preprocessed', 'sentiments')

print('Experiment 3: Effect of N-Grams')
apply_ngram_and_classify(df, 'sentences_preprocessed', 'sentiments', ngram_range=(2, 2))

print('Experiment 4: Effect of BOW vs GLOVE')
# change the sentiments from integers to Strings
sentiments = df['sentiments'].tolist()
df = pd.DataFrame({'sentences_preprocessed': preprocessed_sentences, 'sentiments': sentiments})
bow_and_classify(df.copy(), 'sentences_preprocessed', 'sentiments')
glove_sentiment_analysis(df.copy(), glove_folder)

print('Experiment 5: Effect of LDA')
apply_lda_and_sentiment_analysis(df, 'sentences_preprocessed', 'sentiments', num_topics=5)

print('Experiment 6: Statistical test')
# Accuracies from experiments
experiment_accuracies = [0.8116666666666666, 0.8116666666666666, 0.6216666666666667, 0.8116666666666666, 0.74, 0.8116666666666666]
# Perform one-way ANOVA
f_statistic, p_value_anova = f_oneway([accuracy_baseline], experiment_accuracies)
# Print ANOVA results
print(f"One-way ANOVA p-value: {p_value_anova}")

#visualise data
visualize_total_words_per_sentence_distribution(root_folder)

