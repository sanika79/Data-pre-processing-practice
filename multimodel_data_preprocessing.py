import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten, Concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import requests
from PIL import Image
from io import BytesIO

# 1. Data Loading and Exploration
# Simulate loading data (Replace with actual file loading)
num_samples = 1000  # Adjust this based on the actual dataset size
img_height, img_width = 224, 224  # Input size for EfficientNetB0
num_classes = 191  # Number of product categories

# Simulate image URLs (replace with actual paths or URLs)
image_urls = ["https://example.com/image{}.jpg".format(i) for i in range(num_samples)]

# Simulate textual descriptions
texts = ["This is a description of product {}".format(i) for i in range(num_samples)]

# Simulate categorical labels (product categories)
categories = np.random.randint(0, num_classes, size=(num_samples,))

# Simulate numerical data (e.g., product price)
numerical_data = np.random.rand(num_samples, 1) * 100  # Prices between 0 and 100

# 2. Data Preprocessing
# Encoding categorical labels using one-hot encoding
encoder = OneHotEncoder(sparse=False)
encoded_cats = encoder.fit_transform(categories.reshape(-1, 1))

# Scaling numerical features
scaler = StandardScaler()
scaled_nums = scaler.fit_transform(numerical_data)

# Text vectorization (using Tokenizer and padding)
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
text_sequences = tokenizer.texts_to_sequences(texts)
padded_texts = pad_sequences(text_sequences, maxlen=max_len)

# Image preprocessing function
def preprocess_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize((img_width, img_height))
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Process all images
image_data = np.array([preprocess_image(url) for url in image_urls])

# Train-test split
X_train_text, X_test_text, X_train_cat, X_test_cat, X_train_num, X_test_num, X_train_img, X_test_img, y_train, y_test = train_test_split(
    padded_texts, encoded_cats, scaled_nums, image_data, encoded_cats, test_size=0.2, random_state=42
)

# 3. Model Building
# Text input
text_input = Input(shape=(max_len,), name='text_input')
embedding = Embedding(input_dim=max_words, output_dim=50, input_length=max_len)(text_input)
lstm = LSTM(32)(embedding)

# Categorical input
cat_input = Input(shape=(num_classes,), name='cat_input')
cat_dense = Dense(16, activation='relu')(cat_input)

# Numerical input
num_input = Input(shape=(1,), name='num_input')
num_dense = Dense(16, activation='relu')(num_input)

# Image input using EfficientNetB0
img_input = Input(shape=(img_height, img_width, 3), name='img_input')
base_model = EfficientNetB0(include_top=False, input_tensor=img_input, weights='imagenet')
base_model.trainable = False  # Freeze the base model
img_features = GlobalAveragePooling2D()(base_model.output)

# Concatenate all the features from different modalities
merged = Concatenate()([lstm, cat_dense, num_dense, img_features])

# Add dense layers and output layer for classification
dense_1 = Dense(128, activation='relu')(merged)
dropout = Dropout(0.5)(dense_1)
output = Dense(num_classes, activation='softmax')(dropout)

# Define the model
model = Model(inputs=[text_input, cat_input, num_input, img_input], outputs=output)

# 4. Model Compilation and Training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(
    [X_train_text, X_train_cat, X_train_num, X_train_img], y_train,
    epochs=10,
    batch_size=32,
    validation_data=([X_test_text, X_test_cat, X_test_num, X_test_img], y_test)
)

# 5. Model Evaluation
# Evaluate the model on the test set
loss, accuracy = model.evaluate([X_test_text, X_test_cat, X_test_num, X_test_img], y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Predict on the test set
y_pred = model.predict([X_test_text, X_test_cat, X_test_num, X_test_img])
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluate performance with accuracy score
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_classes)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Confusion matrix
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Optional: Plot training & validation accuracy/loss curves
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
