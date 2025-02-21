import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load JSON Data
# -----------------------------

# Example JSON data
json_data = '''
[
    {"numerical_feature": 1.2, "categorical_feature": "A", "text_feature": "good service", "label": "Positive"},
    {"numerical_feature": 3.4, "categorical_feature": "B", "text_feature": "bad experience", "label": "Negative"},
    {"numerical_feature": 2.1, "categorical_feature": "A", "text_feature": "excellent product", "label": "Positive"},
    {"numerical_feature": 5.6, "categorical_feature": "C", "text_feature": "average", "label": "Neutral"},
    {"numerical_feature": 7.8, "categorical_feature": "B", "text_feature": "poor quality", "label": "Negative"},
    {"numerical_feature": 4.3, "categorical_feature": "C", "text_feature": "best ever", "label": "Positive"},
    {"numerical_feature": 6.7, "categorical_feature": "A", "text_feature": "worst ever", "label": "Negative"},
    {"numerical_feature": 8.9, "categorical_feature": "B", "text_feature": "very nice", "label": "Positive"}
]
'''

# Load JSON data into a Pandas DataFrame
df = pd.DataFrame(json.loads(json_data))

# -----------------------------
# 2. Feature Preprocessing
# -----------------------------

# 2.1 Preprocess Numerical Features
scaler = StandardScaler()
df["numerical_feature"] = scaler.fit_transform(df[["numerical_feature"]])

# 2.2 Preprocess Categorical Features (One-Hot Encoding)
encoder = OneHotEncoder(sparse=False)
categorical_encoded = encoder.fit_transform(df[["categorical_feature"]])
categorical_feature_names = encoder.get_feature_names_out(["categorical_feature"])
df_categorical = pd.DataFrame(categorical_encoded, columns=categorical_feature_names)

# 2.3 Preprocess Textual Features (TF-IDF Vectorization)
vectorizer = TfidfVectorizer(max_features=10)
text_features = vectorizer.fit_transform(df["text_feature"]).toarray()
text_feature_names = vectorizer.get_feature_names_out()
df_text = pd.DataFrame(text_features, columns=text_feature_names)

# 2.4 Encode Labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# -----------------------------
# 3. Prepare Feature Set
# -----------------------------

# Combine all preprocessed features
X = np.hstack([df[["numerical_feature"]].values, df_categorical.values, df_text.values])
feature_names = ["numerical_feature"] + list(categorical_feature_names) + list(text_feature_names)
y = df["label"].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Train GBDT Model (LightGBM)
# -----------------------------

lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)

# -----------------------------
# 5. Feature Importance and Selection
# -----------------------------

# Get feature importance scores
feature_importance = lgb_model.feature_importances_

# Create DataFrame for visualization
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Select top-k features (e.g., top 5)
top_k = 5
selected_features = importance_df["Feature"][:top_k].values
selected_indices = [feature_names.index(f) for f in selected_features]

# -----------------------------
# 6. Retrain Model with Selected Features
# -----------------------------

X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

lgb_model_selected = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model_selected.fit(X_train_selected, y_train)

# -----------------------------
# 7. Evaluate Models
# -----------------------------

y_pred = lgb_model.predict(X_test)
accuracy_full = accuracy_score(y_test, y_pred)

y_pred_selected = lgb_model_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)

print(f"Accuracy with All Features: {accuracy_full:.4f}")
print(f"Accuracy with Selected Features: {accuracy_selected:.4f}")

# Show feature importance
print("\nTop Features Selected:")
print(importance_df.head(top_k))
