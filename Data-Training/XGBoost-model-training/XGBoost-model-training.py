import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_excel("resumes.xlsx")

# Encode categorical data
label_encoders = {}
for column in ["Gender", "Address", "Education", "Certification", "Job Role"]:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Convert experience to months
def extract_experience_in_months(experience_str):
    import re
    years = months = 0
    match = re.search(r"(\d+)\s*years?", str(experience_str))
    if match:
        years = int(match.group(1)) * 12
    match = re.search(r"(\d+)\s*months?", str(experience_str))
    if match:
        months = int(match.group(1))
    return years + months

df["Experience"] = df["Experience"].apply(extract_experience_in_months)

# Features & Target
X = df.drop(columns=["Hired"])  # Features
y = df["Hired"]  # Target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBClassifier()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
