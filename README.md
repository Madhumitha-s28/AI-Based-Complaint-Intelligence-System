# AI-Based Complaint Intelligence System with Prioritization & Decision Support

## Overview
Built an NLP pipeline to classify consumer complaints using SVM and Naive Bayes, benchmarked on held-out test accuracy (54.57% - SVM, 46.49% NB) across 50,000 complaints with integration of VADER sentiment analysis and a rule-based Complaint Severity Index to auto-prioritize complaints (High/Medium/Low). And deployed an interactive Streamlit dashboard for live model comparison.

## Technologies
- Python
- Scikit-learn
- Pandas and NumPy 
- NLTK
- Streamlit

## Features
- TF-IDF feature extraction
- SVM and Naive Bayes models
- Complaint classification
- Sentiment analysis and Severity prediction
- Interactive dashboard

## How to Run

pip install -r requirements.txt

streamlit run app.py
