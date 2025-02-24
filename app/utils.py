import requests
import re
import pickle
import json
import numpy as np
from bs4 import BeautifulSoup
from flask import session
from model import load_ml_model
from app.ml_analysis import ml_model
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Load ML Model & TF-IDF Vectorizer
ml_model, vectorizer = load_ml_model()

def preprocess_text(text):
    """Cleans and prepares text for ML model."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces/newlines
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.strip().lower()  # Convert to lowercase

# ✅ Fact Analysis using ML Model
def analyze_fact(query):
    """Uses ML model with Wikipedia, TOI, and BBC data."""
    sources = ["https://en.wikipedia.org", "https://timesofindia.indiatimes.com", "https://bbc.com"]
    evidence_texts = []

    for source in sources:
        search_results = zenrow_search(f"{query} site:{source}")  # ✅ Use ZenRow for scraping
        if search_results:
            for link in search_results:
                scraped_text = scrape_data(link)
                if scraped_text:
                    evidence_texts.append(scraped_text)

    if not evidence_texts:
        return {"label": "Uncertain", "confidence": 0.0}

    # ✅ Preprocess and Convert to TF-IDF Features
    cleaned_texts = [preprocess_text(text) for text in evidence_texts]
    text_vectors = vectorizer.transform(cleaned_texts)  # ✅ Convert text to numerical features

    # ✅ ML Model Prediction
    predictions = ml_model.predict(text_vectors)

    # 🔹 Debugging: Print model output
    print("DEBUG: ML Model Raw Predictions ->", predictions)

    # ✅ Get Majority Vote
    true_count = np.sum(predictions == "True")
    false_count = np.sum(predictions == "False")

    if true_count > false_count:
        label = "True"
        confidence = (true_count / len(predictions)) * 100
    elif false_count > true_count:
        label = "False"
        confidence = (false_count / len(predictions)) * 100
    else:
        label = "Uncertain"
        confidence = 50

    print("DEBUG: ML Model Final Verdict ->", label, confidence)
    return {"label": label, "confidence": round(confidence, 2)}


# ✅ Google Search API
def google_search(query):
    """Search Google and return the first 10 result links. Store in session."""
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            links = [a["href"] for a in soup.find_all("a", href=True) if "/url?q=" in a["href"]]
            clean_links = [re.sub(r"/url\?q=([^&]+).*", r"\1", link) for link in links]

            session['urls'] = clean_links[:10]  # ✅ Store in session for "View Sources"
            return clean_links[:10]  
    except Exception as e:
        print(f"Google search failed: {e}")
    
    return []

# ✅ ZenRow API Fallback
def zenrow_search(query):
    """Search Google with ZenRow API for general trending sources (for RNN)."""
    ZENROW_API_KEY = "b11aa8a11dce244b879380cacd263344b260216c"
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    zenrow_url = f"https://api.zenrows.com/v1/?apikey={ZENROW_API_KEY}&url={search_url}&js_render=true&premium_proxy=true"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(zenrow_url, headers=headers, timeout=15)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            search_results = soup.select("div.tF2Cxc a")

            # Extract & clean URLs
            clean_links = [re.sub(r"/url\?q=([^&]+).*", r"\1", result["href"]) for result in search_results if result.has_attr("href")]

            session['rnn_urls'] = clean_links[:10]  # ✅ Store for RNN
            return clean_links[:10] if clean_links else []
        else:
            print(f"ZenRow API failed with status code: {response.status_code}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"ZenRow API request failed: {e}")
        return []

# ✅ Web Scraping Function
def scrape_data(url):
    """Fetch meaningful content from a URL for ML and RNN models."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            content = " ".join([p.get_text() for p in paragraphs])
            return content[:1000]  # ✅ Limit text length for relevance
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
    
    return None
