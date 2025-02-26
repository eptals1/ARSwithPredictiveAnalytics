import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_excel("XGBoost-model-training/resumes.xlsx")

# Fill NaN values in Job Requirement
df["Job Requirement"] = df["Job Requirement"].fillna("")

# Dynamically set max_features based on vocabulary size
max_features = min(8, len(set(" ".join(df["Job Requirement"]).split())))
vectorizer = TfidfVectorizer(max_features=max_features)

# Transform job requirements into numerical features
job_requirement_features = vectorizer.fit_transform(df["Job Requirement"]).toarray()
num_features = job_requirement_features.shape[1]  # Get actual feature count

# Convert to DataFrame
job_req_df = pd.DataFrame(job_requirement_features, columns=[f"job_feature_{i}" for i in range(num_features)])

# Merge with original dataset (excluding original text column)
df = df.drop(columns=["Job Requirement"])
df = pd.concat([df, job_req_df], axis=1)

# Save the processed dataset
df.to_csv("processed_resumes.csv", index=False)

print(f"Successfully processed {df.shape[0]} resumes with {num_features} job requirement features!")
