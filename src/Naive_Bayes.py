import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Update path + column names to match your dataset
df = pd.read_csv(r"C:\Users\Madhumitha PC\OneDrive\Desktop\AI ml\data\preprocessed_complaints.csv")

# Example column names (change if needed)
X = df['processed_text']     # complaint text
y = df['Issue']    # issue category


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # helps if classes are imbalanced
)


tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)


y_pred = nb_model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Naive Bayes Accuracy:", round(accuracy * 100, 2), "%\n")

print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("📉 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


pickle.dump(nb_model, open("nb_model.pkl", "wb"))
pickle.dump(tfidf, open("nb_tfidf_vectorizer.pkl", "wb"))  # same one used everywhere
pickle.dump(accuracy, open("nb_accuracy.pkl", "wb"))

print("\n✅ Model, vectorizer, and accuracy saved successfully!")
