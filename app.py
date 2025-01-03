from flask import Flask, render_template, request, session
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

NEWS_API_KEY = "737ac3568c22431da65c1c36332a8650"
NEWS_API_URL = "https://newsapi.org/v2/everything"

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
    tool = language_tool_python.LanguageTool('en-IN')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# Google search function
def google_search(query, num_results=10):
    try:
        urls = []
        for url in search(query, num=num_results, stop=num_results, pause=2):
            urls.append(url)
        return urls
    except Exception as e:
        print(f"Error during Google search: {e}")
        return []

# Fetch news articles using News API
def fetch_news_articles(query, num_results=5):
    try:
        params = {
            "q": query,
            "apiKey": NEWS_API_KEY,
            "pageSize": num_results
        }
        response = requests.get(NEWS_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            urls = [article["url"] for article in articles if "url" in article]
            return urls
        else:
            print(f"News API error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching news articles: {e}")
        return []

# Generate the converse of a claim
def generate_converse(claim):
    if "not" in claim:
        converse = claim.replace("not", "")
    else:
        words = claim.split()
        words.insert(1, "not")
        converse = " ".join(words)
    return converse.strip()

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
    image = image.convert("L")
    image = ImageEnhance.Sharpness(image).enhance(2)
    image = image.point(lambda x: 0 if x < 140 else 255)
    return image

# OCR function for extracting text in multiple languages
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        processed_image = preprocess_image(image)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, lang='eng+hin+mar', config=custom_config)
        return text.strip()
    except Exception as e:
        print(f"Error in OCR process: {e}")
        return ""

@app.route("/", methods=["GET", "POST"])
def index():
    truth_percentage = None
    false_percentage = None
    error = None
    urls = []
    query = None
    corrected_query = None
    label = None
    confidence = None
    converse_query = None
    converse_label = None
    converse_confidence = None

    if request.method == "POST":
        input_type = request.form.get("input_type")  # 'news' or 'other'
        image = request.files.get("image")
        user_query = request.form.get("query")

        if image and not user_query:
            try:
                user_query = extract_text_from_image(image)
                print("Extracted text:", user_query)
            except Exception as e:
                print(f"Error extracting text from image: {e}")
                error = "Failed to process the uploaded image."
                return render_template("index.html", error=error)

        if not user_query:
            error = "Please enter a claim or upload an image."
            return render_template("index.html", error=error)

        corrected_query = correct_grammar(user_query)
        converse_query = generate_converse(corrected_query)

        if input_type == "news":
            urls = fetch_news_articles(corrected_query, num_results=5)
            if not urls:  # Fallback to Google search
                urls = google_search(corrected_query, num_results=5)
        else:
            urls = google_search(corrected_query, num_results=5)

        if not urls:
            error = "No search results found."
            return render_template("index.html", error=error)

        session['urls'] = urls

        label, confidence = predict_label(model, tokenizer, corrected_query)
        converse_label, converse_confidence = predict_label(model, tokenizer, converse_query)

        truth_percentage = round((confidence + (100 - converse_confidence)) / 2, 2)
        false_percentage = round(100 - truth_percentage, 2)

        is_true = truth_percentage > 50

        return render_template(
            "index.html",
            query=user_query,
            corrected_query=corrected_query,
            label=label,
            confidence=confidence,
            converse_query=converse_query,
            converse_label=converse_label,
            converse_confidence=converse_confidence,
            truth_percentage=truth_percentage,
            false_percentage=false_percentage,
            is_true=is_true,
            urls=urls,
            error=error
        )

    return render_template(
        "index.html",
        truth_percentage=truth_percentage,
        false_percentage=false_percentage,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
