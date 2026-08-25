import pandas as pd  
# pandas → used for loading and manipulating tabular datasets

from sklearn.model_selection import train_test_split  
# train_test_split → splits dataset into training and testing sets
# This allows evaluation of model performance on unseen data

from sklearn.svm import LinearSVC  
# LinearSVC → Support Vector Machine classifier optimized
# for large text datasets with linear decision boundaries

from sklearn.metrics import accuracy_score, classification_report  
# accuracy_score → calculates overall prediction accuracy
# classification_report → provides precision, recall, and F1-score for each class

from sklearn.feature_extraction.text import TfidfVectorizer  
# TfidfVectorizer → converts text data into numerical feature vectors
# using TF-IDF weighting for machine learning models

from sklearn.metrics import accuracy_score
import pickle


# Load dataset containing the processed complaint text
df = pd.read_csv(r"C:\Users\Madhumitha PC\OneDrive\Desktop\AI ml\data\preprocessed_complaints.csv")

# Check dataset size before training
print("Original Dataset:", df.shape)


    
# Count number of complaints in each issue category
counts = df['Issue'].value_counts()

# Keep only issue categories having at least 200 complaints
# Rare classes may cause poor model learning and imbalance
df = df[df['Issue'].isin(counts[counts >= 200].index)]

print("After removing rare classes:", df.shape)


# X → input text used for model training
X = df['processed_text']

# y → target variable representing complaint issue category
y = df['Issue']


tfidf = TfidfVectorizer(

    max_features=15000,
    # Limits vocabulary size to most important 15000 words
    # Helps reduce dimensionality and memory usage

    stop_words='english',
    # Removes common English words that carry little meaning

    ngram_range=(1,2),
    # Uses both single words (unigrams) and word pairs (bigrams)
    # Helps capture context like "credit card" or "loan payment"

    min_df=5,
    # Ignore words appearing in fewer than 5 documents
    # Removes extremely rare terms

    max_df=0.85,
    # Ignore words appearing in more than 85% of documents
    # Removes overly common terms that add little value

    sublinear_tf=True
    # Applies logarithmic scaling to term frequency
    # Helps reduce the effect of very frequent words
)

# Convert complaint text into TF-IDF feature matrix
X_tfidf = tfidf.fit_transform(X)

print("TF-IDF Shape:", X_tfidf.shape)


X_train, X_test, y_train, y_test = train_test_split(

    X_tfidf,
    y,

    test_size=0.2,
    # 20% data used for testing model performance

    random_state=42,
    # Ensures reproducibility of results

    stratify=y
    # Maintains same class distribution in train and test sets
)


model = LinearSVC(

    C=1.5,
    # Regularization parameter controlling model complexity

    class_weight='balanced',
    # Automatically adjusts weights for imbalanced classes

    max_iter=5000
    # Allows more iterations for better convergence
)

# Train the SVM model using training data
model.fit(X_train, y_train)


# Predict issue categories for test data
pred = model.predict(X_test)


# Calculate overall classification accuracy
print("\nAccuracy:", accuracy_score(y_test, pred))

# Detailed performance metrics for each issue class
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

import pickle

# Save trained model
pickle.dump(model, open("svm_issue_model.pkl", "wb"))

# Save TF-IDF vectorizer
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))

print("Model and TF-IDF saved successfully!")



accuracy = accuracy_score(y_test,pred)
# Save accuracy
pickle.dump(accuracy, open("accuracy.pkl", "wb"))
