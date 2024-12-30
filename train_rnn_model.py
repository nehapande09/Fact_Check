import json
import tensorflow as tf
from keras.src.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np



# Load the dataset
texts = []
labels = []

with open("train.jsonl", "r") as file:
    for line in file:
        data = json.loads(line)
        texts.append(data['claim'])  # Use 'claim' as the key for input text
        labels.append(1 if data['label'] == "SUPPORTS" else 0)  # Binary label: SUPPORTS -> 1, else -> 0

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Padding sequences
max_length = 100  # Adjust based on your needs
data = pad_sequences(sequences, maxlen=max_length)

# Convert labels to numpy array
labels = np.array(labels)

# Build the RNN model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_length),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  #Binary
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=30, batch_size=32, validation_split=0.2)

# Save the model and tokenizer
model.save("rnn_model.h5")
with open("tokenizer.pickle", "wb") as handle:
    import pickle
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and tokenizer saved successfully.")
