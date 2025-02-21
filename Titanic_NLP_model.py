import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#print("dataframe", df.head(5)) 
# print(df['Name'].iloc[0])

# Drop rows with missing values
df.dropna(inplace=True)

# -----------------------------
# 2. Identify Data Types
# -----------------------------
text_column = "Name"  # Using passenger names as a text feature
categorical_columns = ["Sex", "Embarked"]  # Categorical features
numerical_columns = ["Age", "Fare"]  # Numerical features
target_column = "Survived"  # Target variable

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df[text_column] = df[text_column].astype(str).apply(clean_text)


# -----------------------------
# 4. Feature Engineering
# -----------------------------
# Encode Categorical Variables
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Scale Numerical Data
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Convert Text to TF-IDF Features
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df[text_column]).toarray()

# Prepare Final Features
X_categorical = df[categorical_columns].values

X_numerical = df[numerical_columns].values

# Concatenate All Features
X = np.hstack((X_text, X_categorical, X_numerical))
y = df[target_column].values

# -----------------------------
# 5. Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")






