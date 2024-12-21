from flask import Flask, render_template, request, redirect, url_for, session
from googlesearch import search
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import requests
from bs4 import BeautifulSoup
import language_tool_python  # Import language tool for grammar correction

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load the trained model and tokenizer
def load_model_and_tokenizer():
    model = load_model("rnn_model.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Preprocessing function for text
def preprocess_text(text):
    """
    Cleans and tokenizes text.
    """
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and trim
    return text

# Grammar correction function using language_tool_python
def correct_grammar(text):
    """
    Corrects grammar mistakes using language_tool_python.
    """
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)  # Get grammar mistakes
    corrected_text = language_tool_python.utils.correct(text, matches)  # Apply corrections
    return corrected_text

# Google search function
def google_search(query, num_results=5):
    """
    Perform a Google search and return top N result URLs.
    """
    try:
        urls = []
        for url in search(query, num=num_results, stop=num_results, pause=2):
            urls.append(url)
        return urls
    except Exception as e:
        print(f"Error during Google search: {e}")
        return []

# Predict function
def predict_label(model, tokenizer, claim):
    """
    Predicts the label and confidence for a given claim.
    """
    sequence = tokenizer.texts_to_sequences([claim])
    max_length = 100  # Ensure consistency with training
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)[0][0]
    label = "SUPPORTS" if prediction > 0.5 else "REFUTES"
    confidence = round(prediction * 100, 2)  # Convert to percentage
    return label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the user query from the form
        user_query = request.form.get("query")
        if not user_query:
            return render_template("index.html", error="Please enter a claim.")

        # Correct grammar of the user's input
        corrected_query = correct_grammar(user_query)

        # Perform Google search
        urls = google_search(corrected_query, num_results=5)
        if not urls:
            return render_template("index.html", error="No search results found.")

        # Save the URLs to the session for access on the sources page
        session['urls'] = urls

        # Predict label and confidence using the corrected claim
        label, confidence = predict_label(model, tokenizer, corrected_query)

        # Render result
        return render_template(
            "index.html",
            query=user_query,
            corrected_query=corrected_query,
            label=label,
            confidence=confidence
        )

    return render_template("index.html")

@app.route("/sources")
def sources():
    # Retrieve URLs from the session
    urls = session.get('urls', [])
    return render_template("sources.html", urls=urls)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
