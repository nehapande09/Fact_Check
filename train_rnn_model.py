from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import json

# Load and preprocess dataset
texts = []
labels = []
with open("train.jsonl", "r") as file:
    for line in file:
        data = json.loads(line)
        texts.append(data['claim'])
        labels.append(1 if data['label'] == "SUPPORTS" else 0)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=100)
labels = np.array(labels)

# ✅ Define the RNN model (Make sure this is present)
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Train the model
model.fit(data, labels, epochs=3, batch_size=32, validation_split=0.2)

# ✅ Save the model and tokenizer
model.save("rnn_model.keras")  # Use the recommended Keras format
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Model and tokenizer saved successfully.")
