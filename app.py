from flask import Flask, request, render_template
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import re
import tensorflow as tf
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Load pre-trained RNN model and tokenizer
rnn_model = tf.keras.models.load_model("rnn_model.h5")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Text preprocessing function
def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Function to scrape data from a URL
def scrape_website(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return preprocess_text(response.text)
        else:
            return ""
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# Function to prepare input for the RNN model
def prepare_input(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    return padded_sequences

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/search', methods=['POST'])
def search_and_process():
    user_input = request.form.get('query')
    if not user_input:
        return render_template('result.html', prediction="No query provided. Please try again.")

    # Perform Google search
    search_results = list(search(user_input, num_results=5))

    # Scrape and preprocess data from search results
    scraped_data = " ".join([scrape_website(url) for url in search_results])

    # Prepare data for RNN
    model_input = prepare_input(scraped_data)

    # Make prediction
    prediction = rnn_model.predict(model_input)
    predicted_label = "Supported" if prediction[0][0] > 0.5 else "Refuted"
    accuracy_score = prediction[0][0] * 100

    return render_template(
        'result.html',
        prediction=f"Prediction: {predicted_label} (Score: {accuracy_score:.2f}%)"
    )

if __name__ == '__main__':
    app.run(debug=True)
