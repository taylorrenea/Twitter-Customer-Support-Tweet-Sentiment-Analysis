# Twitter-Customer-Support-Tweet-Sentiment-Analysis
# The following code is meant to be used in order to analyze a set of Twitter customer support oriented tweets for their overall sentiment.
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler  # Import for oversampling

# Download NLTK stopwords
nltk.download('stopwords')

# Define preprocessing function
def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()

    # Remove URLs
    tweet = re.sub(r'http\\S+|www\\S+|https\\S+', '', tweet, flags=re.MULTILINE)

    # Remove mentions
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)

    # Remove special characters and punctuation
    tweet = re.sub(r'[^A-Za-z\s]', '', tweet)

    # Tokenize tweet while preserving spaces
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=False)
    tokens = tokenizer.tokenize(tweet)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a string while preserving spaces
    preprocessed_tweet = ' '.join(filtered_tokens)

    return preprocessed_tweet


# Load the dataset
file_name = 'twitter_support_data.csv'
file_path = os.path.join(os.getcwd(), file_name)
twitter_data = pd.read_csv(file_path)

# Apply preprocessing to the 'text' column
twitter_data['preprocessed_text'] = twitter_data['text'].apply(preprocess_tweet)

# Vectorize the preprocessed text using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(twitter_data['preprocessed_text'])
y = twitter_data['text']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random oversampling to balance the data
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

# Train a Naive Bayes classifier on the resampled data
nb_classifier = MultinomialNB()
nb_classifier.fit(X_resampled, y_resampled)

# Predict sentiment on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred, zero_division=1))

# Print some examples of predicted tweets along with their actual labels
print("\nSome examples of predicted tweets along with their actual labels:")
for i in range(10):  # Adjust the range as needed
    print(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")
