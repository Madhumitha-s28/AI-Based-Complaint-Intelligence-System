# pandas → used for handling tabular datasets
import pandas as pd

# numpy → useful for numerical operations
import numpy as np

# re → used for regular expression operations (text pattern removal)
import re

# string → provides list of punctuation characters
import string

# nltk → Natural Language Toolkit for NLP preprocessing
import nltk

# Import stopwords list from NLTK
from nltk.corpus import stopwords

# Import lemmatizer to convert words to base form
from nltk.stem import WordNetLemmatizer


# stopwords → common words like "the", "is", "and"
# These words usually don't carry meaningful information
nltk.download('stopwords')

# wordnet → lexical database required for lemmatization
nltk.download('wordnet')



# Read training dataset from local system
train = pd.read_csv(r"C:\Users\Madhumitha PC\OneDrive\Desktop\AI ml\data\complaints_train.csv")

# Read test dataset
test = pd.read_csv(r"C:\Users\Madhumitha PC\OneDrive\Desktop\AI ml\data\complaints_test.csv")


# Clean column names by removing leading/trailing spaces
# This prevents column mismatch errors later
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()


# Display dataset shapes to understand dataset size
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# Display first few rows to inspect dataset
print(train.head())


# The original dataset contains many columns.
# For this project we only keep the columns
# relevant for complaint analysis and classification.

columns_needed = [
    "Product",                       # Product category (e.g., Credit Card)
    "Sub-product",                   # Sub category
    "Issue",                         # Main issue type
    "Sub-issue",                     # Detailed issue
    "Consumer complaint narrative",  # Actual complaint text
    "Company",                       # Company involved
    "State",                         # Customer state
    "Submitted via",                 # Submission channel (web, phone, etc.)
    "Company response to consumer",  # Company's action
    "Timely response?"               # Whether response was timely
]

# Keep only selected columns
train = train[columns_needed]
test = test[columns_needed]


# "Consumer complaint narrative" is a long column name.
# Rename it to "Complaint" for easier coding.

train = train.rename(columns={"Consumer complaint narrative": "Complaint"})
test = test.rename(columns={"Consumer complaint narrative": "Complaint"})


# Some rows contain empty complaint narratives.
# These rows cannot be used for text analysis.

train = train.dropna(subset=["Complaint"])
test = test.dropna(subset=["Complaint"])


# Duplicate complaints can bias the machine learning model.
# Therefore duplicates based on complaint text are removed.

train = train.drop_duplicates(subset=["Complaint"])
test = test.drop_duplicates(subset=["Complaint"])



# Load English stopwords
# Example stopwords: the, is, at, on, etc.
stop_words = set(stopwords.words('english'))


# Lemmatization converts words into their base form
# Example:
#   running → run
#   complaints → complaint
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    # Convert text to lowercase
    # Example: "BANK Complaint" → "bank complaint"
    text = str(text).lower()

    # Remove URLs if present
    # Example: http://example.com
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove numbers from text
    # Example: "loan 123 denied" → "loan denied"
    text = re.sub(r"\d+", "", text)

    # Remove punctuation characters
    # Example: "loan denied!" → "loan denied"
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    # Example: "loan    denied" → "loan denied"
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize words
    words = text.split()

    # Remove stopwords and apply lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # Join cleaned words back into a sentence
    return " ".join(words)



# Apply the cleaning function to every complaint
# This creates a new column "Cleaned_Complaint"

train["Cleaned_Complaint"] = train["Complaint"].apply(clean_text)
test["Cleaned_Complaint"] = test["Complaint"].apply(clean_text)


# Complaints with very few words may not contain
# enough information for machine learning models.

# Keep complaints with more than 3 words
train = train[train["Cleaned_Complaint"].str.split().str.len() > 3]
test = test[test["Cleaned_Complaint"].str.split().str.len() > 3]


# After removing rows, indices become inconsistent.
# Reset index to maintain proper order.

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)


# Save processed datasets to new CSV files
# These files will be used for feature extraction
# (TF-IDF / Word embeddings) and model training.

train.to_csv(r"C:\Users\Madhumitha PC\OneDrive\Desktop\AI ml\data\cleaned_train.csv", index=False)
test.to_csv(r"C:\Users\Madhumitha PC\OneDrive\Desktop\AI ml\data\cleaned_test.csv", index=False)


# Print dataset size after cleaning
print("Cleaned Train Shape:", train.shape)
print("Cleaned Test Shape:", test.shape)

# Display sample cleaned data
print(train.head())
