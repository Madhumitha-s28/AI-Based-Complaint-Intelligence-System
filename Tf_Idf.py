# Import required libraries

import pandas as pd
# pandas → used for loading, manipulating, and analyzing datasets

from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer → converts text data into numerical vectors
# using Term Frequency–Inverse Document Frequency weighting


# Load dataset containing cleaned and processed complaint text
df = pd.read_csv(r"C:\Users\Madhumitha PC\OneDrive\Desktop\AI ml\data\preprocessed_complaints.csv")

# Display dataset size (number of rows and columns)
print("Dataset Shape:", df.shape)


# 'processed_text' column → contains fully cleaned complaint text
# This text will be converted into numerical features
X = df['processed_text']



tfidf = TfidfVectorizer(

    max_features=5000,
    # Limits vocabulary to top 5000 important words
    # Helps reduce dimensionality and memory usage

    stop_words='english',
    # Removes common English stopwords such as "the", "is", "and"
    # These words usually do not add meaningful information

    ngram_range=(1,2)
    # Uses both:
    # (1,1) → single words (unigrams)
    # (1,2) → word pairs (bigrams)
    # Example: "credit card", "loan payment"
)


X_tfidf = tfidf.fit_transform(X)

# Display shape of TF-IDF matrix
# Rows → number of complaints
# Columns → number of extracted features (words)
print("TF-IDF Matrix Shape:", X_tfidf.shape)


# Convert sparse TF-IDF matrix into a pandas DataFrame
# This is useful for viewing or analyzing feature values

tfidf_df = pd.DataFrame(

    X_tfidf.toarray(),
    # Converts sparse matrix to dense array

    columns=tfidf.get_feature_names_out()
    # Extracts the actual word features used in TF-IDF
)

# Display first few rows of the TF-IDF feature dataset
print(tfidf_df.head())


# Save extracted TF-IDF features to a CSV file
# This file can be used later for model training or analysis

tfidf_df.to_csv("tfidf_features.csv", index=False)

print("TF-IDF features saved as 'tfidf_features.csv'")
