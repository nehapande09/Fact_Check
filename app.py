from flask import Flask, render_template, request, redirect, url_for, session
from googlesearch import search
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import requests
from bs4 import BeautifulSoup
import language_tool_python
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract  # OCR library

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the trained model and tokenizer
def load_model_and_tokenizer():
    model = load_model("rnn_model.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Preprocessing function for text
def preprocess_text(text):
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower().strip()
    return text

# Grammar correction function
def correct_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# Google search function
def google_search(query, num_results=5):
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
    sequence = tokenizer.texts_to_sequences([claim])
    max_length = 100
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)[0][0]
    label = "SUPPORTS" if prediction > 0.5 else "REFUTES"
    confidence = round(prediction * 100, 2)
    return label, confidence

# Preprocessing image before OCR
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")
    # Enhance sharpness
    image = ImageEnhance.Sharpness(image).enhance(2)
    # Apply thresholding
    image = image.point(lambda x: 0 if x < 140 else 255)
    return image

# OCR function for extracting text in multiple languages
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        processed_image = preprocess_image(image)
        # Use Tesseract with language support for Hindi, English, and Marathi
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, lang='eng+hin+mar', config=custom_config)
        return text.strip()
    except Exception as e:
        print(f"Error in OCR process: {e}")
        return ""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image is uploaded
        image = request.files.get("image")
        user_query = request.form.get("query")

        if image and not user_query:
            # Extract text from the uploaded image
            try:
                user_query = extract_text_from_image(image)
                print("Extracted text:", user_query)  # Debugging log
            except Exception as e:
                print(f"Error extracting text from image: {e}")
                return render_template("index.html", error="Failed to process the uploaded image.")

        if not user_query:
            return render_template("index.html", error="Please enter a claim or upload an image.")

        # Correct grammar of the user's input
        corrected_query = correct_grammar(user_query)

        # Perform Google search
        urls = google_search(corrected_query, num_results=5)
        if not urls:
            return render_template("index.html", error="No search results found.")

        session['urls'] = urls

        # Predict label and confidence using the corrected claim
        label, confidence = predict_label(model, tokenizer, corrected_query)

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

if __name__ == "__main__":
    app.run(debug=True)
