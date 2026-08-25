
# TEXT PREPROCESSING 

import pandas as pd        # Used to load and manipulate dataset tables
import re                  # Used for removing unwanted patterns from text
import nltk #Natural Language Toolkit (NLTK) used for natural language processing (NLP) and computation..
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLP resources required for preprocessing
# stopwords → removes common meaningless words
# wordnet → required for lemmatization (better -> good)
nltk.download('stopwords')
nltk.download('wordnet')


# Load the dataset that was already reduced and balanced
# This dataset is smaller and more suitable for ML training
df = pd.read_csv(r"C:\Users\Madhumitha PC\OneDrive\Desktop\AI ml\data\reduced_train_dataset.csv")

# Check dataset size to understand how many complaints will be processed
print("Original Dataset Shape:", df.shape)



# Remove rows where complaint text is missing
# Missing text cannot be used for NLP processing
df = df.dropna(subset=['Cleaned_Complaint'])

# Remove rows where complaint text exists but contains only spaces
# Such rows do not provide meaningful information for the model
df = df[df['Cleaned_Complaint'].astype(str).str.strip() != ""]

print("After removing empty complaints:", df.shape)



# Stopwords list helps remove very common words
# These words appear frequently but do not help classification
stop_words = set(stopwords.words('english'))

# Lemmatizer converts words to their root form
# This reduces vocabulary size and improves model learning
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # Convert text to lowercase
    # Ensures 'Bank' and 'bank' are treated as the same word
    text = str(text).lower()

    # Remove numbers because they usually do not help in complaint classification
    text = re.sub(r'\d+', '', text)

    # Remove punctuation symbols to simplify the text
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenization: split sentence into individual words
    # This allows word-level filtering and processing
    words = text.split()

    # Remove stopwords and apply lemmatization
    # This keeps only meaningful base words for ML models
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Join processed words back into a sentence
    return " ".join(words)



# Apply the preprocessing function to all complaint texts
# Store the final cleaned text in a new column used for ML models
df['processed_text'] = df['Cleaned_Complaint'].apply(preprocess_text)



# Display original complaint and processed version
# Helps verify that preprocessing worked correctly
print("\nSample processed complaints:")
print(df[['Cleaned_Complaint','processed_text']].head())


# Save the processed dataset so it can be used for
# feature extraction (TF-IDF) and model training
df.to_csv("preprocessed_complaints.csv", index=False)

print("\nPreprocessed dataset saved as 'preprocessed_complaints.csv'")
