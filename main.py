# Import required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset files
train_data = pd.read_csv('atis_intents_train.csv')
test_data = pd.read_csv('atis_intents_test.csv')
full_data = pd.read_csv('atis_intents.csv')

# Display the first few rows of each dataset to check the structure
print("Training Data:")
print(train_data.head())

print("\nTesting Data:")
print(test_data.head())

# Display column names in the training and testing datasets
print("Training Data Columns:", train_data.columns)
print("Testing Data Columns:", test_data.columns)

# Extract columns explicitly
X_train = train_data.iloc[:, 1]  # Queries (2nd column, index 1)
y_train = train_data.iloc[:, 0]  # Intents (1st column, index 0)

X_test = test_data.iloc[:, 1]  # Queries (2nd column, index 1)
y_test = test_data.iloc[:, 0]  # Intents (1st column, index 0)

# Label encoding for intents
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Tokenize and convert text to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)  # Fit tokenizer on training data only

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences for uniform input size
X_train_pad = pad_sequences(X_train_seq, padding='post')
X_test_pad = pad_sequences(X_test_seq, padding='post')

# Display shapes
print(f"Training data shape: {X_train_pad.shape}")
print(f"Test data shape: {X_test_pad.shape}")
print(f"Number of unique intents: {len(label_encoder.classes_)}")

# Define model parameters
vocab_size = len(tokenizer.word_index) + 1  # Total vocabulary size (+1 for padding token)
max_len = X_train_pad.shape[1]  # Sequence length from padded data
num_classes = len(label_encoder.classes_)  # Number of unique intents

# Verify parameters
print(f"Vocab Size: {vocab_size}, Max Sequence Length: {max_len}, Number of Classes: {num_classes}")

# Build the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.build(input_shape=(None, max_len))  # Explicitly define the input shape
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Display the model summary
model.summary()

# Train the model
history = model.fit(
    X_train_pad,
    y_train,
    validation_split=0.2,  # 20% of training data for validation
    epochs=10,
    batch_size=32,
    verbose=1
)

# Save the model for future use
model.save('intent_classification_model.keras')

# Predict on test data
y_pred = model.predict(X_test_pad)
y_pred_classes = y_pred.argmax(axis=1)  # Get predicted class indices

# Calculate test accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test Accuracy: {accuracy:.2f}")

# Generate classification report
report = classification_report(
    y_test,
    y_pred_classes,
    target_names=label_encoder.classes_
)
print(report)

# Function to predict intent from a user query
def predict_intent(query):
    # Preprocess the query
    seq = tokenizer.texts_to_sequences([query])
    pad_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    
    # Predict intent
    prediction = model.predict(pad_seq)
    predicted_class = prediction.argmax(axis=1)[0]
    intent = label_encoder.inverse_transform([predicted_class])[0]
    
    return intent

# Example usage
user_query = "show me flights from New York to Los Angeles"
predicted_intent = predict_intent(user_query)
print(f"Predicted Intent: {predicted_intent}")
