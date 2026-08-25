import pandas as pd   # pandas is used for dataset manipulation and analysis


# Read the dataset generated after text cleaning and preprocessing
df = pd.read_csv(r"C:\Users\Madhumitha PC\OneDrive\Desktop\AI ml\data\cleaned_train.csv")

# Display original dataset size (rows, columns)
print("Original Dataset Shape:", df.shape)


# 'Issue' column → represents the complaint category
# (example: Billing issues, Loan problems, Credit card complaints)

# Count number of complaints per issue category
issue_counts = df['Issue'].value_counts()

# Select issues that have at least 500 complaints
# This ensures sufficient training data for each category
valid_issues = issue_counts[issue_counts >= 500].index

# Keep only rows where the Issue is in the valid issue list
df = df[df['Issue'].isin(valid_issues)]

print("After removing rare issues:", df.shape)



# 'Cleaned_Complaint' column → contains preprocessed complaint text
# Very short complaints may lack useful information for ML models

df = df[df['Cleaned_Complaint'].str.split().str.len() > 10]

# Extremely long complaints may introduce noise and increase processing cost
df = df[df['Cleaned_Complaint'].str.split().str.len() < 150]

print("After filtering complaint length:", df.shape)



# Group dataset based on the 'Issue' category
# This ensures each issue category is processed separately
df = (
    df.groupby('Issue', group_keys=False)

    # Sample complaints from each issue category
    # min(len(x),2000) → ensures maximum 2000 complaints per issue
    # random_state → ensures reproducibility of random sampling
    .apply(lambda x: x.sample(min(len(x), 2000), random_state=42))
)

# Reset index after grouping and sampling
df = df.reset_index(drop=True)

print("After balancing dataset:", df.shape)



# Randomly shuffle rows so that complaints from different
# issue categories are mixed before training ML models
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



# Save the processed dataset which will be used
# for feature extraction and machine learning training
df.to_csv("reduced_train_dataset.csv", index=False)

print("Final Reduced Dataset Shape:", df.shape)




# Print number of complaints per issue category
# to verify dataset balanceo
print("\nIssue Distribution:\n")
print(df['Issue'].value_counts())
