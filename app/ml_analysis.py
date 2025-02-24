import requests
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import io
import base64
import wikipedia

# Load pre-trained ML model
try:
    with open("app/models/fact_check_model.pkl", "rb") as model_file:
        ml_model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading ML model: {e}")
    ml_model = None

# Load TF-IDF Vectorizer
try:
    with open("app/models/tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    print(f"Error loading vectorizer: {e}")
    vectorizer = None

# Trusted sources for ML fact-checking
TRUSTED_SOURCES = {
    "Wikipedia": "https://en.wikipedia.org/wiki/",
    "Times of India": "https://timesofindia.indiatimes.com/topic/",
    "BBC News": "https://www.bbc.co.uk/search?q="
}

def fetch_wikipedia_summary(query):
    """Fetches a summary from Wikipedia for ML Analysis."""
    try:
        summary = wikipedia.summary(query, sentences=3)
        return [{"source": "Wikipedia", "summary": summary}]
    except wikipedia.exceptions.DisambiguationError as e:
        return [{"source": "Wikipedia", "summary": f"Multiple results found: {e.options[:5]}"}]
    except wikipedia.exceptions.PageError:
        return [{"source": "Wikipedia", "summary": "No Wikipedia page found for this query."}]

def scrape_source(url):
    """Scrapes text content from the given URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text() for p in paragraphs[:3]])  # Limit text to the first 3 paragraphs
            return text.strip()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    return ""

def fetch_news_articles(query):
    """Fetches news articles related to a query from TOI and BBC for ML Analysis."""
    articles = []
    
    for source, base_url in TRUSTED_SOURCES.items():
        if source == "Wikipedia":
            continue  # Skip Wikipedia as it's handled separately
        
        search_url = base_url + query.replace(" ", "_")
        content = scrape_source(search_url)
        if content:
            articles.append({"source": source, "summary": content})
    
    return articles

def fetch_evidence(query):
    """Fetches evidence from trusted sources for ML Analysis."""
    wikipedia_data = fetch_wikipedia_summary(query)
    news_data = fetch_news_articles(query)
    return wikipedia_data + news_data

def analyze_fact(query, model=ml_model):
    """
    Analyzes the given query using ML and evidence from trusted sources.
    Returns a dictionary with fact-checking analysis results.
    """
    if model is None or vectorizer is None:
        return {"error": "ML model or vectorizer not loaded."}
    
    evidence_texts = fetch_evidence(query)
    if not evidence_texts:
        return {
            "label": "Uncertain",
            "confidence": 50.0,
            "explanation": "No sufficient evidence found from trusted sources."
        }
    
    combined_evidence = " ".join([entry["summary"] for entry in evidence_texts])
    query_vectorized = vectorizer.transform([query + " " + combined_evidence])
    
    prediction = model.predict(query_vectorized)[0]
    confidence = float(model.decision_function(query_vectorized)[0]) * 100
    
    if confidence < 60:
        label = "Uncertain"
    else:
        label = "True" if prediction == 1 else "False"
    
    return {
        "label": label,
        "confidence": round(confidence, 2),
        "explanation": f"The claim is classified as {label} with {round(confidence, 2)}% confidence based on available evidence.",
        "sources": evidence_texts
    }


