import pandas as pd
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Load Data from Mixed Formats
# ------------------------------

# Load CSV Data
csv_data = pd.read_csv("data.csv")  # Ensure it has 'text', 'category', 'numerical_value', 'label' columns

# Load JSON Data
with open("data.json", "r") as f:
    json_data = json.load(f)
json_df = pd.DataFrame(json_data)  # Convert JSON to DataFrame

# Load TXT Data (Each line is a text sample)
with open("data.txt", "r") as f:
    txt_data = f.readlines()
txt_df = pd.DataFrame({"text": txt_data})  # Convert to DataFrame

# Merge all data
df = pd.concat([csv_data, json_df, txt_df], ignore_index=True)

# ------------------------------
# 2. Data Preprocessing
# ------------------------------

# Encode Labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Encode Categorical Feature (e.g., category column)
category_encoder = LabelEncoder()
df["category_encoded"] = category_encoder.fit_transform(df["category"].fillna("unknown"))

# Scale Numerical Data
scaler = StandardScaler()
df["numerical_value"] = scaler.fit_transform(df[["numerical_value"]].fillna(0))

# Tokenize Text
MAX_WORDS = 10000  # Vocabulary size
MAX_LEN = 128  # Maximum length of text sequences

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"].astype(str))

df["tokenized_text"] = tokenizer.texts_to_sequences(df["text"].astype(str))
df["padded_text"] = list(pad_sequences(df["tokenized_text"], maxlen=MAX_LEN, padding="post"))

# Convert to NumPy Arrays
X_text = np.array(df["padded_text"].tolist())
X_category = np.array(df["category_encoded"].values)
X_numerical = np.array(df["numerical_value"].values).reshape(-1, 1)
y = np.array(df["label"].values)

# Split Data
X_train_text, X_test_text, X_train_cat, X_test_cat, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, X_category, X_numerical, y, test_size=0.2, random_state=42
)

# ------------------------------
# 3. Define Multimodal NLP Model
# ------------------------------

# Text Input (LSTM)
text_input = keras.layers.Input(shape=(MAX_LEN,), name="text_input")
embedding = keras.layers.Embedding(MAX_WORDS, 128, input_length=MAX_LEN)(text_input)
lstm = keras.layers.LSTM(64, return_sequences=False)(embedding)

# Categorical Input (Embedding)
category_input = keras.layers.Input(shape=(1,), name="category_input")
category_emb = keras.layers.Embedding(len(category_encoder.classes_), 8)(category_input)
category_flat = keras.layers.Flatten()(category_emb)

# Numerical Input (Dense Layer)
numerical_input = keras.layers.Input(shape=(1,), name="numerical_input")
numerical_dense = keras.layers.Dense(8, activation="relu")(numerical_input)

# Concatenate Features
concat = keras.layers.Concatenate()([lstm, category_flat, numerical_dense])
dense1 = keras.layers.Dense(64, activation="relu")(concat)
dropout = keras.layers.Dropout(0.3)(dense1)
output = keras.layers.Dense(len(label_encoder.classes_), activation="softmax")(dropout)

# Define Model
model = keras.models.Model(inputs=[text_input, category_input, numerical_input], outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Model Summary
model.summary()

# ------------------------------
# 4. Train Model
# ------------------------------

EPOCHS = 5
BATCH_SIZE = 32

history = model.fit(
    [X_train_text, X_train_cat, X_train_num],
    y_train,
    validation_data=([X_test_text, X_test_cat, X_test_num], y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ------------------------------
# 5. Evaluate Model
# ------------------------------

test_loss, test_acc = model.evaluate([X_test_text, X_test_cat, X_test_num], y_test)
print(f"Test Accuracy: {test_acc:.4f}")
