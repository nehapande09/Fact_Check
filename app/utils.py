import re
import requests
from googlesearch import search
from bs4 import BeautifulSoup
import language_tool_python
from PIL import Image, ImageEnhance
import pytesseract
from tensorflow.keras.preprocessing.sequence import pad_sequences

def google_search(query, num_results=10):
    try:
        return [url for url in search(query, num=num_results, stop=num_results, pause=2)]
    except Exception as e:
        print(f"Error during Google search: {e}")
        return []

def fetch_news_articles(query, num_results=5):
    try:
        params = {
            "q": query,
            "apiKey": "737ac3568c22431da65c1c36332a8650",
            "pageSize": num_results
        }
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        if response.status_code == 200:
            data = response.json()
            return [article["url"] for article in data.get("articles", []) if "url" in article]
        else:
            print(f"News API error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching news articles: {e}")
        return []

def fetch_historical_facts(query, num_results=5):
    try:
        params = {
            "q": query,
            "apiKey": "vJBp9kFqRPKh7pSIimKJSKvRRjzjs0G7",
            "pageSize": num_results
        }
        response = requests.get("https://api.nytimes.com/svc/archive/v1/2018/9.json?api-key=vJBp9kFqRPKh7pSIimKJSKvRRjzjs0G7", params=params)
        if response.status_code == 200:
            data = response.json()
            return [article["url"] for article in data.get("articles", []) if "url" in article]

        else:

            print(f"Archive API error: {response.status_code} - {response.text}")

            return []

    except Exception as e:

        print(f"Error fetching historical data: {e}")

        return []



def fetch_current_data(query, num_results=5):
    try:
        api_key = 'your_currents_api_key'  # Replace with your Currents API key
        url = 'https://api.currentsapi.services/v1/search'
        params = {
            'apiKey': api_key,
            'keywords': query,
            'language': 'en',
            'limit': num_results
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return [article['url'] for article in data.get('news', []) if 'url' in article]
        else:
            print(f'Currents API error: {response.status_code} - {response.text}')
            return []
    except Exception as e:
        print(f'Error fetching current data: {e}')
        return []



def correct_grammar(text):
    tool = language_tool_python.LanguageTool('en-IN')
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

def generate_converse(claim):
    if "not" in claim:
        return claim.replace("not", "")
    words = claim.split()
    words.insert(1, "not")
    return " ".join(words).strip()

def predict_label(model, tokenizer, claim):
    sequence = tokenizer.texts_to_sequences([claim])
    max_length = 100
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)[0][0]
    label = "SUPPORTS" if prediction > 0.5 else "REFUTES"
    return label, round(prediction * 100, 2)

def preprocess_image(image):
    image = image.convert("L")
    return ImageEnhance.Sharpness(image).enhance(2).point(lambda x: 0 if x < 140 else 255)

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        processed_image = preprocess_image(image)
        return pytesseract.image_to_string(processed_image, lang='eng+hin+mar', config=r'--oem 3 --psm 6').strip()
    except Exception as e:
        print(f"Error in OCR process: {e}")
        return ""

